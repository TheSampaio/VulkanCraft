#include "vulkancraft/rendering/VulkanRenderer.hpp"

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

#include "vulkancraft/core/Camera.hpp"
#include "vulkancraft/rendering/AtlasMipmaps.hpp"
#include "vulkancraft/rendering/CascadeShadow.hpp"
#include "vulkancraft/rendering/CelestialCycle.hpp"
#include "vulkancraft/rendering/Frustum.hpp"
#include "vulkancraft/world/WorldMesher.hpp"
#include "vulkancraft/world/StreamingWorld.hpp"

namespace vulkancraft::rendering {
namespace {

#if defined(VULKANCRAFT_ENABLE_VALIDATION) && !defined(NDEBUG)
constexpr bool validation_requested = true;
#else
constexpr bool validation_requested = false;
#endif

constexpr std::uint32_t shadow_resolution = 2048U;
constexpr VkFormat scene_color_format = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr std::uint32_t atlas_tile_size = 32U;
constexpr std::uint32_t atlas_mip_level_count = 6U;
constexpr std::array<const char*, 1> validation_layers{"VK_LAYER_KHRONOS_validation"};
constexpr std::array<const char*, 1> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

/** Throws a readable exception when a Vulkan call fails. */
void require_success(const VkResult result, const std::string_view operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with Vulkan result " + std::to_string(result));
    }
}

/** Routes validation diagnostics to the standard error stream. */
VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    const VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    const VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    static_cast<void>(type);
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "Vulkan validation: " << callback_data->pMessage << '\n';
    }
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT && user_data != nullptr) {
        static_cast<std::atomic_bool*>(user_data)->store(true, std::memory_order_relaxed);
    }
    return VK_FALSE;
}

/** Returns the debug messenger configuration shared by instance creation. */
[[nodiscard]] VkDebugUtilsMessengerCreateInfoEXT debug_messenger_info(void* user_data) noexcept {
    VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debug_callback;
    info.pUserData = user_data;
    return info;
}

/** Holds graphics and presentation queue family selections. */
struct QueueFamilies {
    std::optional<std::uint32_t> graphics;
    std::optional<std::uint32_t> present;

    /** Returns true when both required queue families were found. */
    [[nodiscard]] bool complete() const noexcept {
        return graphics.has_value() && present.has_value();
    }
};

/** Owns temporary swapchain capability query results. */
struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> present_modes;
};

/** Mirrors the std140 frame uniform block used by both shader pipelines. */
struct alignas(16) FrameUniforms {
    glm::mat4 view{1.0F};
    glm::mat4 projection{1.0F};
    std::array<glm::mat4, cascade_count> light_view_projection{};
    glm::vec4 cascade_splits{};
    glm::vec4 light_direction{};
    glm::vec4 camera_position{};
    glm::vec4 light_color_intensity{};
    glm::vec4 sky_color{};
    glm::vec4 sun_direction{};
    glm::vec4 animation_data{};
    std::array<glm::uvec4, 16> console_text{};
    glm::uvec4 console_state{};
};

struct alignas(16) CelestialPushConstants {
    glm::vec4 direction_and_half_size{};
    glm::vec4 color{};
    glm::vec4 world_right{};
    glm::vec4 world_up{};
};

/** Owns one independently replaceable GPU mesh batch. */
struct MeshBufferSet {
    VkBuffer vertex_buffer{VK_NULL_HANDLE};
    VkDeviceMemory vertex_memory{VK_NULL_HANDLE};
    VkBuffer index_buffer{VK_NULL_HANDLE};
    VkDeviceMemory index_memory{VK_NULL_HANDLE};
    std::uint32_t index_count{};
    VkBuffer transparent_index_buffer{VK_NULL_HANDLE};
    VkDeviceMemory transparent_index_memory{VK_NULL_HANDLE};
    std::uint32_t transparent_index_count{};
    VkBuffer cloud_index_buffer{VK_NULL_HANDLE};
    VkDeviceMemory cloud_index_memory{VK_NULL_HANDLE};
    std::uint32_t cloud_index_count{};
    VkBuffer shadow_index_buffer{VK_NULL_HANDLE};
    VkDeviceMemory shadow_index_memory{VK_NULL_HANDLE};
    std::uint32_t shadow_index_count{};
};

/** Stores one dynamic chunk batch in a single suballocated Vulkan buffer. */
struct PackedMeshBufferSet {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    VkDeviceSize vertex_offset{};
    VkDeviceSize index_offset{};
    VkDeviceSize transparent_index_offset{};
    VkDeviceSize cloud_index_offset{};
    VkDeviceSize shadow_index_offset{};
    std::uint32_t index_count{};
    std::uint32_t transparent_index_count{};
    std::uint32_t cloud_index_count{};
    std::uint32_t shadow_index_count{};
};

/** Shares one uploaded batch between independently drawn chunk ranges. */
struct MeshBufferOwner {
    VkDevice device{VK_NULL_HANDLE};
    PackedMeshBufferSet buffers;

    /** Releases the complete batch while the renderer device is still alive. */
    ~MeshBufferOwner() {
        if (device == VK_NULL_HANDLE) {
            return;
        }
        vkDestroyBuffer(device, buffers.buffer, nullptr);
        vkFreeMemory(device, buffers.memory, nullptr);
    }
};

/** Selects one chunk range from a shared uploaded mesh batch. */
struct ChunkMeshBinding {
    std::shared_ptr<MeshBufferOwner> owner;
    world::MeshDrawRange range;
};

/** Locates consecutive indexed indirect commands in the persistent command buffer. */
struct IndirectDrawSpan {
    VkDeviceSize offset{};
    std::uint32_t count{};
};

/** Owns staging resources until one batched mesh transfer completes. */
struct PendingMeshUpload {
    MeshBufferSet destination;
    VkBuffer vertex_staging{VK_NULL_HANDLE};
    VkDeviceMemory vertex_staging_memory{VK_NULL_HANDLE};
    VkBuffer index_staging{VK_NULL_HANDLE};
    VkDeviceMemory index_staging_memory{VK_NULL_HANDLE};
    VkBuffer transparent_staging{VK_NULL_HANDLE};
    VkDeviceMemory transparent_staging_memory{VK_NULL_HANDLE};
    VkBuffer cloud_staging{VK_NULL_HANDLE};
    VkDeviceMemory cloud_staging_memory{VK_NULL_HANDLE};
    VkBuffer shadow_staging{VK_NULL_HANDLE};
    VkDeviceMemory shadow_staging_memory{VK_NULL_HANDLE};
    VkDeviceSize vertex_size{};
    VkDeviceSize index_size{};
    VkDeviceSize transparent_index_size{};
    VkDeviceSize cloud_index_size{};
    VkDeviceSize shadow_index_size{};
};

/** Retains asynchronous packed-upload resources until their transfer fence signals. */
struct PendingPackedUpload {
    VkBuffer staging{VK_NULL_HANDLE};
    VkDeviceMemory staging_memory{VK_NULL_HANDLE};
    VkCommandBuffer command{VK_NULL_HANDLE};
    VkFence fence{VK_NULL_HANDLE};
};

enum class PipelineKind {
    opaque,
    wireframe,
    transparent,
    cloud,
    depth_prepass,
    cloud_depth_prepass,
    shadow,
    sky,
    celestial,
    postprocess,
};

static_assert(sizeof(FrameUniforms) % 16U == 0U);

/** Reads one binary SPIR-V file produced by the shader build target. */
[[nodiscard]] std::vector<std::uint32_t> read_spirv(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open shader: " + path.string());
    }
    const auto end = file.tellg();
    if (end <= 0 || end % static_cast<std::streamoff>(sizeof(std::uint32_t)) != 0) {
        throw std::runtime_error("Shader bytecode is invalid: " + path.string());
    }

    std::vector<std::uint32_t> words(
        static_cast<std::size_t>(end) / sizeof(std::uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), end);
    if (!file) {
        throw std::runtime_error("Could not read shader: " + path.string());
    }
    return words;
}

/** Returns the invariant full-height world bounds of one chunk column. */
[[nodiscard]] AxisAlignedBoundingBox chunk_bounds(
    const world::ChunkCoordinate coordinate,
    const float elapsed_seconds) noexcept {
    constexpr float chunk_size = 32.0F;
    constexpr float minimum_y = -64.0F;
    constexpr float maximum_y = 320.0F;
    constexpr float maximum_cloud_speed = 0.13F;
    constexpr glm::vec2 conservative_wind_direction{0.925F, 0.382F};
    const float minimum_x = static_cast<float>(coordinate.x) * chunk_size;
    const float minimum_z = static_cast<float>(coordinate.z) * chunk_size;
    const glm::vec2 maximum_cloud_displacement =
        conservative_wind_direction * std::max(elapsed_seconds, 0.0F) * maximum_cloud_speed;
    return {
        .minimum = {minimum_x, minimum_y, minimum_z},
        .maximum = {
            minimum_x + chunk_size + maximum_cloud_displacement.x,
            maximum_y,
            minimum_z + chunk_size + maximum_cloud_displacement.y,
        },
    };
}

}  // namespace

class VulkanRenderer::Impl final {
public:
    /** Initializes every Vulkan resource required by the renderer. */
    Impl(
        GLFWwindow* window,
        const world::WorldMesh& mesh,
        const std::span<const world::ChunkCoordinate> chunk_coordinates)
        : window_(window) {
        if (window_ == nullptr || mesh.vertices.empty() || mesh.indices.empty() ||
            chunk_coordinates.size() != mesh.draw_ranges.size()) {
            throw std::invalid_argument("Renderer requires a window and a non-empty mesh");
        }
        try {
            create_instance();
            create_debug_messenger();
            create_surface();
            select_physical_device();
            create_device();
            create_swapchain();
            create_command_pool();
            create_descriptor_resources();
            create_atlas_resources();
            create_mesh_buffers(mesh);
            create_indirect_resources(mesh.draw_ranges.size());
            active_chunk_coordinates_.assign(
                chunk_coordinates.begin(), chunk_coordinates.end());
            create_shadow_resources();
            create_depth_resources();
            create_scene_color_resources();
            update_descriptor_set();
            create_pipelines();
            allocate_command_buffer();
            create_synchronization();
        } catch (...) {
            cleanup();
            throw;
        }
    }

    /** Releases resources in the inverse order of their creation. */
    ~Impl() {
        cleanup();
    }

    /** Renders and presents one synchronized frame. */
    void draw_frame(
        const core::Camera& camera,
        const float world_time_seconds,
        const float animation_seconds,
        const bool underwater) {
        if (framebuffer_resized_) {
            recreate_swapchain();
        }

        require_success(
            vkWaitForFences(device_, 1, &frame_fence_, VK_TRUE, std::numeric_limits<std::uint64_t>::max()),
            "Waiting for the frame fence");
        release_completed_packed_uploads(false);

        std::uint32_t image_index = 0;
        const VkResult acquire_result = vkAcquireNextImageKHR(
            device_,
            swapchain_,
            std::numeric_limits<std::uint64_t>::max(),
            image_available_,
            VK_NULL_HANDLE,
            &image_index);
        if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreate_swapchain();
            return;
        }
        if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
            require_success(acquire_result, "Acquiring a swapchain image");
        }

        update_uniforms(camera, world_time_seconds, animation_seconds, underwater);
        record_commands(image_index);
        require_success(vkResetFences(device_, 1, &frame_fence_), "Resetting the frame fence");

        VkSemaphoreSubmitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        wait_info.semaphore = image_available_;
        wait_info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        wait_info.value = 0;

        VkCommandBufferSubmitInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        command_info.commandBuffer = command_buffer_;

        VkSemaphoreSubmitInfo signal_info{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal_info.semaphore = render_finished_[image_index];
        signal_info.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        signal_info.value = 0;

        VkSubmitInfo2 submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit_info.waitSemaphoreInfoCount = 1;
        submit_info.pWaitSemaphoreInfos = &wait_info;
        submit_info.commandBufferInfoCount = 1;
        submit_info.pCommandBufferInfos = &command_info;
        submit_info.signalSemaphoreInfoCount = 1;
        submit_info.pSignalSemaphoreInfos = &signal_info;
        require_success(vkQueueSubmit2(graphics_queue_, 1, &submit_info, frame_fence_), "Submitting a frame");

        VkPresentInfoKHR present_info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = &render_finished_[image_index];
        present_info.swapchainCount = 1;
        present_info.pSwapchains = &swapchain_;
        present_info.pImageIndices = &image_index;
        const VkResult present_result = vkQueuePresentKHR(present_queue_, &present_info);
        if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR ||
            framebuffer_resized_) {
            recreate_swapchain();
        } else {
            require_success(present_result, "Presenting a swapchain image");
        }
        ++frame_sequence_;
    }

    /** Defers resize work until Vulkan reaches a safe frame boundary. */
    void notify_framebuffer_resized() noexcept {
        framebuffer_resized_ = true;
    }

    /** Replaces immutable geometry after the streaming window changes. */
    void update_world_mesh(const world::WorldMesh& mesh) {
        if (mesh.vertices.empty() || mesh.indices.empty()) {
            throw std::invalid_argument("A streamed world mesh must not be empty");
        }
        require_success(
            vkWaitForFences(
                device_,
                1,
                &frame_fence_,
                VK_TRUE,
                std::numeric_limits<std::uint64_t>::max()),
            "Waiting for the active frame before a world mesh update");
        destroy_chunk_overrides();
        vkDestroyBuffer(device_, shadow_index_buffer_, nullptr);
        vkFreeMemory(device_, shadow_index_memory_, nullptr);
        vkDestroyBuffer(device_, cloud_index_buffer_, nullptr);
        vkFreeMemory(device_, cloud_index_memory_, nullptr);
        vkDestroyBuffer(device_, transparent_index_buffer_, nullptr);
        vkFreeMemory(device_, transparent_index_memory_, nullptr);
        vkDestroyBuffer(device_, index_buffer_, nullptr);
        vkFreeMemory(device_, index_memory_, nullptr);
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vkFreeMemory(device_, vertex_memory_, nullptr);
        index_buffer_ = VK_NULL_HANDLE;
        index_memory_ = VK_NULL_HANDLE;
        transparent_index_buffer_ = VK_NULL_HANDLE;
        transparent_index_memory_ = VK_NULL_HANDLE;
        cloud_index_buffer_ = VK_NULL_HANDLE;
        cloud_index_memory_ = VK_NULL_HANDLE;
        shadow_index_buffer_ = VK_NULL_HANDLE;
        shadow_index_memory_ = VK_NULL_HANDLE;
        vertex_buffer_ = VK_NULL_HANDLE;
        vertex_memory_ = VK_NULL_HANDLE;
        index_count_ = 0;
        transparent_index_count_ = 0;
        cloud_index_count_ = 0;
        shadow_index_count_ = 0;
        draw_ranges_.clear();
        create_mesh_buffers(mesh);
    }

    /** Replaces one cached chunk without rebuilding or uploading the other chunks. */
    void update_world_chunk(const std::size_t chunk_index, const world::WorldMesh& mesh) {
        const bool has_indices = !mesh.indices.empty() || !mesh.transparent_indices.empty() ||
                                 !mesh.cloud_indices.empty() || !mesh.shadow_indices.empty();
        if (chunk_index >= draw_ranges_.size() || mesh.vertices.empty() || !has_indices) {
            throw std::invalid_argument("A chunk mesh update is invalid");
        }
        require_success(
            vkWaitForFences(
                device_,
                1,
                &frame_fence_,
                VK_TRUE,
                std::numeric_limits<std::uint64_t>::max()),
            "Waiting for the active frame before a chunk mesh update");
        auto owner = std::make_shared<MeshBufferOwner>();
        owner->device = device_;
        owner->buffers = upload_packed_mesh_buffer(mesh);
        chunk_overrides_[chunk_index] = ChunkMeshBinding{
            .owner = std::move(owner),
            .range = {
                .first_index = 0,
                .index_count = static_cast<std::uint32_t>(mesh.indices.size()),
                .first_transparent_index = 0,
                .transparent_index_count =
                    static_cast<std::uint32_t>(mesh.transparent_indices.size()),
                .first_cloud_index = 0,
                .cloud_index_count = static_cast<std::uint32_t>(mesh.cloud_indices.size()),
                .first_shadow_index = 0,
                .shadow_index_count = static_cast<std::uint32_t>(mesh.shadow_indices.size()),
            },
        };
    }

    /** Reuses overlapping chunks and uploads only newly visible coordinates. */
    void update_streamed_chunks(
        const std::span<const world::ChunkCoordinate> chunk_coordinates,
        const std::span<const std::size_t> replacement_indices,
        const world::WorldMesh& replacement_mesh) {
        if (chunk_coordinates.size() != draw_ranges_.size()) {
            throw std::invalid_argument("A streamed chunk window has an invalid size");
        }
        require_success(
            vkWaitForFences(
                device_,
                1,
                &frame_fence_,
                VK_TRUE,
                std::numeric_limits<std::uint64_t>::max()),
            "Waiting for the active frame before remapping streamed chunks");

        std::vector<std::optional<ChunkMeshBinding>> replacements(draw_ranges_.size());
        std::vector<std::size_t> replacement_base_ranges(
            draw_ranges_.size(), std::numeric_limits<std::size_t>::max());
        std::vector<bool> reused(active_chunk_coordinates_.size(), false);
        std::vector<std::size_t> replacement_range_for_slot(
            draw_ranges_.size(), std::numeric_limits<std::size_t>::max());
        for (std::size_t range_index = 0; range_index < replacement_indices.size();
             ++range_index) {
            const std::size_t slot = replacement_indices[range_index];
            if (slot >= replacement_range_for_slot.size() ||
                replacement_range_for_slot[slot] != std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument("A streamed replacement slot is invalid");
            }
            replacement_range_for_slot[slot] = range_index;
        }
        std::shared_ptr<MeshBufferOwner> replacement_owner;
        if (replacement_mesh.draw_ranges.size() != replacement_indices.size() ||
            (replacement_indices.empty() != replacement_mesh.vertices.empty())) {
            throw std::invalid_argument("A streamed upload batch does not match replacement chunks");
        }
        if (!replacement_indices.empty()) {
            replacement_owner = std::make_shared<MeshBufferOwner>();
            replacement_owner->device = device_;
            replacement_owner->buffers = upload_packed_mesh_buffer(replacement_mesh);
        }

        const auto coordinate_key = [](const world::ChunkCoordinate coordinate) {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(coordinate.x)) << 32U) |
                   static_cast<std::uint32_t>(coordinate.z);
        };
        std::unordered_map<std::uint64_t, std::size_t> old_slot_by_coordinate;
        old_slot_by_coordinate.reserve(active_chunk_coordinates_.size());
        for (std::size_t old_index = 0; old_index < active_chunk_coordinates_.size(); ++old_index) {
            old_slot_by_coordinate.emplace(
                coordinate_key(active_chunk_coordinates_[old_index]), old_index);
        }
        for (std::size_t new_index = 0; new_index < chunk_coordinates.size(); ++new_index) {
            const auto existing = old_slot_by_coordinate.find(
                coordinate_key(chunk_coordinates[new_index]));
            const std::size_t replacement_range = replacement_range_for_slot[new_index];
            if (replacement_range != std::numeric_limits<std::size_t>::max()) {
                if (existing != old_slot_by_coordinate.end()) {
                    reused[existing->second] = true;
                }
                replacements[new_index] = ChunkMeshBinding{
                    .owner = replacement_owner,
                    .range = replacement_mesh.draw_ranges[replacement_range],
                };
            } else if (existing != old_slot_by_coordinate.end()) {
                const std::size_t old_index = existing->second;
                reused[old_index] = true;
                replacement_base_ranges[new_index] = base_range_for_slot_[old_index];
                if (chunk_overrides_[old_index].has_value()) {
                    replacements[new_index] = *chunk_overrides_[old_index];
                    chunk_overrides_[old_index].reset();
                }
            } else {
                throw std::logic_error("An entering streamed chunk has no replacement range");
            }
        }
        for (std::size_t old_index = 0; old_index < chunk_overrides_.size(); ++old_index) {
            if (!reused[old_index]) {
                chunk_overrides_[old_index].reset();
            }
        }
        chunk_overrides_ = std::move(replacements);
        base_range_for_slot_ = std::move(replacement_base_ranges);
        active_chunk_coordinates_.assign(chunk_coordinates.begin(), chunk_coordinates.end());
    }

    /** Switches between the prebuilt solid and line polygon pipelines. */
    void set_wireframe_enabled(const bool enabled) noexcept {
        wireframe_enabled_ = enabled;
    }

    /** Stores console text for the next uniform-buffer update. */
    void set_console_overlay(
        const bool visible,
        const std::string_view text) noexcept {
        console_visible_ = visible;
        console_text_.assign(text.substr(0, 64U));
    }

    /** Waits for all device work when a device exists. */
    void wait_idle() const noexcept {
        if (device_ != VK_NULL_HANDLE) {
            static_cast<void>(vkDeviceWaitIdle(device_));
        }
    }

    /** Reports whether the validation callback observed any error severity message. */
    [[nodiscard]] bool has_validation_errors() const noexcept {
        return validation_error_seen_.load(std::memory_order_relaxed);
    }

private:
    /** Creates a Vulkan 1.3 instance and enables optional validation. */
    void create_instance() {
        std::uint32_t loader_version = VK_API_VERSION_1_0;
        require_success(vkEnumerateInstanceVersion(&loader_version), "Querying the Vulkan loader version");
        if (loader_version < VK_API_VERSION_1_3) {
            throw std::runtime_error("VulkanCraft requires a Vulkan 1.3 loader");
        }

        VkApplicationInfo application_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application_info.pApplicationName = "VulkanCraft";
        application_info.applicationVersion = VK_MAKE_API_VERSION(
            0,
            VULKANCRAFT_VERSION_MAJOR,
            VULKANCRAFT_VERSION_MINOR,
            VULKANCRAFT_VERSION_PATCH);
        application_info.pEngineName = "VulkanCraft";
        application_info.engineVersion = application_info.applicationVersion;
        application_info.apiVersion = VK_API_VERSION_1_3;

        std::uint32_t glfw_extension_count = 0;
        const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
        if (glfw_extensions == nullptr || glfw_extension_count == 0) {
            throw std::runtime_error("GLFW did not provide Vulkan instance extensions");
        }
        std::vector<const char*> extensions(
            glfw_extensions, glfw_extensions + glfw_extension_count);
        if (validation_requested && validation_layers_available()) {
            validation_enabled_ = true;
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create_info.pApplicationInfo = &application_info;
        create_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debug_info{};
        if (validation_enabled_) {
            create_info.enabledLayerCount = static_cast<std::uint32_t>(validation_layers.size());
            create_info.ppEnabledLayerNames = validation_layers.data();
            debug_info = debug_messenger_info(&validation_error_seen_);
            create_info.pNext = &debug_info;
        }
        require_success(vkCreateInstance(&create_info, nullptr, &instance_), "Creating the Vulkan instance");
    }

    /** Checks that every requested validation layer is installed. */
    [[nodiscard]] bool validation_layers_available() const {
        std::uint32_t count = 0;
        require_success(vkEnumerateInstanceLayerProperties(&count, nullptr), "Listing validation layers");
        std::vector<VkLayerProperties> properties(count);
        require_success(
            vkEnumerateInstanceLayerProperties(&count, properties.data()),
            "Reading validation layers");
        return std::ranges::all_of(validation_layers, [&properties](const char* requested) {
            return std::ranges::any_of(properties, [requested](const VkLayerProperties& available) {
                return std::strcmp(requested, available.layerName) == 0;
            });
        });
    }

    /** Creates the debug callback when validation is active. */
    void create_debug_messenger() {
        if (!validation_enabled_) {
            return;
        }
        const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create == nullptr) {
            throw std::runtime_error("The debug utils extension is unavailable");
        }
        auto info = debug_messenger_info(&validation_error_seen_);
        require_success(create(instance_, &info, nullptr, &debug_messenger_), "Creating the debug messenger");
    }

    /** Creates the GLFW presentation surface. */
    void create_surface() {
        require_success(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_), "Creating the window surface");
    }

    /** Chooses the best device that supports the complete renderer contract. */
    void select_physical_device() {
        std::uint32_t device_count = 0;
        require_success(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr), "Counting Vulkan devices");
        if (device_count == 0) {
            throw std::runtime_error("No Vulkan-capable GPU was found");
        }
        std::vector<VkPhysicalDevice> devices(device_count);
        require_success(
            vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
            "Listing Vulkan devices");

        int best_score = -1;
        for (const VkPhysicalDevice candidate : devices) {
            const auto families = find_queue_families(candidate);
            const auto swapchain_support = query_swapchain_support(candidate);
            if (!families.complete() || !device_extensions_available(candidate) ||
                swapchain_support.formats.empty() || swapchain_support.present_modes.empty()) {
                continue;
            }

            VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &features13;
            vkGetPhysicalDeviceFeatures2(candidate, &features);
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_3 || features13.dynamicRendering != VK_TRUE ||
                features13.synchronization2 != VK_TRUE ||
                features.features.fillModeNonSolid != VK_TRUE ||
                features.features.multiDrawIndirect != VK_TRUE) {
                continue;
            }

            const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 1000 : 100;
            if (score > best_score) {
                best_score = score;
                physical_device_ = candidate;
                queue_families_ = families;
            }
        }
        if (physical_device_ == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "No GPU provides Vulkan 1.3, dynamic rendering, synchronization2, wireframe, and swapchain support");
        }
    }

    /** Finds graphics and surface presentation queues for a device. */
    [[nodiscard]] QueueFamilies find_queue_families(const VkPhysicalDevice device) const {
        std::uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
        std::vector<VkQueueFamilyProperties> properties(count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());

        QueueFamilies families;
        for (std::uint32_t index = 0; index < count; ++index) {
            if ((properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U) {
                families.graphics = index;
            }
            VkBool32 present_supported = VK_FALSE;
            require_success(
                vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface_, &present_supported),
                "Querying surface support");
            if (present_supported == VK_TRUE) {
                families.present = index;
            }
            if (families.complete()) {
                break;
            }
        }
        return families;
    }

    /** Verifies the required device extension set. */
    [[nodiscard]] bool device_extensions_available(const VkPhysicalDevice device) const {
        std::uint32_t count = 0;
        require_success(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr),
                        "Counting device extensions");
        std::vector<VkExtensionProperties> properties(count);
        require_success(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, properties.data()),
                        "Listing device extensions");
        return std::ranges::all_of(device_extensions, [&properties](const char* requested) {
            return std::ranges::any_of(properties, [requested](const VkExtensionProperties& available) {
                return std::strcmp(requested, available.extensionName) == 0;
            });
        });
    }

    /** Queries swapchain surface capabilities, formats, and presentation modes. */
    [[nodiscard]] SwapchainSupport query_swapchain_support(const VkPhysicalDevice device) const {
        SwapchainSupport support;
        require_success(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &support.capabilities),
                        "Querying surface capabilities");
        std::uint32_t count = 0;
        require_success(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, nullptr),
                        "Counting surface formats");
        support.formats.resize(count);
        if (count > 0) {
            require_success(vkGetPhysicalDeviceSurfaceFormatsKHR(
                                device, surface_, &count, support.formats.data()),
                            "Listing surface formats");
        }
        require_success(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, nullptr),
                        "Counting presentation modes");
        support.present_modes.resize(count);
        if (count > 0) {
            require_success(vkGetPhysicalDeviceSurfacePresentModesKHR(
                                device, surface_, &count, support.present_modes.data()),
                            "Listing presentation modes");
        }
        return support;
    }

    /** Creates queues and a logical device with Vulkan 1.3 features. */
    void create_device() {
        const std::set<std::uint32_t> unique_families{
            queue_families_.graphics.value(), queue_families_.present.value()};
        constexpr float priority = 1.0F;
        std::vector<VkDeviceQueueCreateInfo> queue_infos;
        for (const std::uint32_t family : unique_families) {
            VkDeviceQueueCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            info.queueFamilyIndex = family;
            info.queueCount = 1;
            info.pQueuePriorities = &priority;
            queue_infos.push_back(info);
        }

        VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        features13.synchronization2 = VK_TRUE;
        features13.dynamicRendering = VK_TRUE;
        VkPhysicalDeviceFeatures device_features{};
        device_features.fillModeNonSolid = VK_TRUE;
        device_features.multiDrawIndirect = VK_TRUE;

        VkDeviceCreateInfo create_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        create_info.pNext = &features13;
        create_info.pEnabledFeatures = &device_features;
        create_info.queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size());
        create_info.pQueueCreateInfos = queue_infos.data();
        create_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
        create_info.ppEnabledExtensionNames = device_extensions.data();
        require_success(vkCreateDevice(physical_device_, &create_info, nullptr, &device_),
                        "Creating the logical device");
        vkGetDeviceQueue(device_, queue_families_.graphics.value(), 0, &graphics_queue_);
        vkGetDeviceQueue(device_, queue_families_.present.value(), 0, &present_queue_);
    }

    /** Chooses an unorm surface format so capture layers may request storage usage. */
    [[nodiscard]] static VkSurfaceFormatKHR choose_surface_format(
        const std::vector<VkSurfaceFormatKHR>& formats) noexcept {
        const auto preferred = std::ranges::find_if(formats, [](const VkSurfaceFormatKHR& format) {
            return format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                   format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        if (preferred != formats.end()) {
            return *preferred;
        }
        const auto alternate = std::ranges::find_if(formats, [](const VkSurfaceFormatKHR& format) {
            return format.format == VK_FORMAT_R8G8B8A8_UNORM &&
                   format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        return alternate != formats.end() ? *alternate : formats.front();
    }

    /** Chooses mailbox presentation when available and FIFO otherwise. */
    [[nodiscard]] static VkPresentModeKHR choose_present_mode(
        const std::vector<VkPresentModeKHR>& modes) noexcept {
        return std::ranges::find(modes, VK_PRESENT_MODE_MAILBOX_KHR) != modes.end()
                   ? VK_PRESENT_MODE_MAILBOX_KHR
                   : VK_PRESENT_MODE_FIFO_KHR;
    }

    /** Chooses the current framebuffer extent within surface limits. */
    [[nodiscard]] VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& capabilities) const {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        return {
            std::clamp(static_cast<std::uint32_t>(width),
                       capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width),
            std::clamp(static_cast<std::uint32_t>(height),
                       capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height),
        };
    }

    /** Creates the swapchain and one color view per acquired image. */
    void create_swapchain() {
        const auto support = query_swapchain_support(physical_device_);
        const auto format = choose_surface_format(support.formats);
        const auto present_mode = choose_present_mode(support.present_modes);
        const auto extent = choose_extent(support.capabilities);
        std::uint32_t image_count = support.capabilities.minImageCount + 1U;
        if (support.capabilities.maxImageCount > 0U) {
            image_count = std::min(image_count, support.capabilities.maxImageCount);
        }

        VkSwapchainCreateInfoKHR create_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        create_info.surface = surface_;
        create_info.minImageCount = image_count;
        create_info.imageFormat = format.format;
        create_info.imageColorSpace = format.colorSpace;
        create_info.imageExtent = extent;
        create_info.imageArrayLayers = 1;
        create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        const std::array<std::uint32_t, 2> families{
            queue_families_.graphics.value(), queue_families_.present.value()};
        if (families[0] != families[1]) {
            create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            create_info.queueFamilyIndexCount = static_cast<std::uint32_t>(families.size());
            create_info.pQueueFamilyIndices = families.data();
        } else {
            create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        create_info.preTransform = support.capabilities.currentTransform;
        create_info.compositeAlpha = choose_composite_alpha(support.capabilities.supportedCompositeAlpha);
        create_info.presentMode = present_mode;
        create_info.clipped = VK_TRUE;
        require_success(vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_),
                        "Creating the swapchain");

        require_success(vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr),
                        "Counting swapchain images");
        swapchain_images_.resize(image_count);
        require_success(vkGetSwapchainImagesKHR(
                            device_, swapchain_, &image_count, swapchain_images_.data()),
                        "Listing swapchain images");
        swapchain_format_ = format.format;
        swapchain_extent_ = extent;
        swapchain_image_initialized_.assign(image_count, false);

        swapchain_views_.reserve(image_count);
        for (const VkImage image : swapchain_images_) {
            swapchain_views_.push_back(create_image_view(
                image, swapchain_format_, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_VIEW_TYPE_2D, 0, 1));
        }
    }

    /** Selects a supported surface alpha composition mode. */
    [[nodiscard]] static VkCompositeAlphaFlagBitsKHR choose_composite_alpha(
        const VkCompositeAlphaFlagsKHR supported) noexcept {
        constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4> preferences{
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        };
        const auto selected = std::ranges::find_if(preferences, [supported](const auto option) {
            return (supported & option) != 0U;
        });
        return selected != preferences.end() ? *selected : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    }

    /** Creates the resettable graphics command pool. */
    void create_command_pool() {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = queue_families_.graphics.value();
        require_success(vkCreateCommandPool(device_, &info, nullptr, &command_pool_),
                        "Creating the command pool");
    }

    /** Creates uniform and shadow descriptor layouts, storage, and sampler. */
    void create_descriptor_resources() {
        VkDescriptorSetLayoutBinding uniform_binding{};
        uniform_binding.binding = 0;
        uniform_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniform_binding.descriptorCount = 1;
        uniform_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding shadow_binding{};
        shadow_binding.binding = 1;
        shadow_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadow_binding.descriptorCount = 1;
        shadow_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding atlas_binding{};
        atlas_binding.binding = 2;
        atlas_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        atlas_binding.descriptorCount = 1;
        atlas_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding scene_binding{};
        scene_binding.binding = 3;
        scene_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        scene_binding.descriptorCount = 1;
        scene_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding depth_binding{};
        depth_binding.binding = 4;
        depth_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        depth_binding.descriptorCount = 1;
        depth_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        const std::array bindings{
            uniform_binding, shadow_binding, atlas_binding, scene_binding, depth_binding};

        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        require_success(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &descriptor_set_layout_),
                        "Creating the descriptor layout");

        create_buffer(
            sizeof(FrameUniforms),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniform_buffer_,
            uniform_memory_);
        require_success(vkMapMemory(device_, uniform_memory_, 0, sizeof(FrameUniforms), 0, &uniform_mapped_),
                        "Mapping the frame uniform buffer");

        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = VK_FILTER_NEAREST;
        sampler_info.minFilter = VK_FILTER_NEAREST;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        sampler_info.maxLod = 1.0F;
        require_success(vkCreateSampler(device_, &sampler_info, nullptr, &shadow_sampler_),
                        "Creating the shadow sampler");

        sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        sampler_info.maxLod = static_cast<float>(atlas_mip_level_count - 1U);
        require_success(vkCreateSampler(device_, &sampler_info, nullptr, &atlas_sampler_),
                        "Creating the block atlas sampler");

        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.maxLod = 0.0F;
        require_success(vkCreateSampler(device_, &sampler_info, nullptr, &scene_sampler_),
                        "Creating the scene-color sampler");
        sampler_info.magFilter = VK_FILTER_NEAREST;
        sampler_info.minFilter = VK_FILTER_NEAREST;
        require_success(vkCreateSampler(device_, &sampler_info, nullptr, &depth_sampler_),
                        "Creating the scene-depth sampler");

        const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
        }};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        require_success(vkCreateDescriptorPool(device_, &pool_info, nullptr, &descriptor_pool_),
                        "Creating the descriptor pool");

        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = descriptor_pool_;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &descriptor_set_layout_;
        require_success(vkAllocateDescriptorSets(device_, &allocation, &descriptor_set_),
                        "Allocating the descriptor set");
    }

    /** Creates one host-visible buffer used to batch all base-chunk draw submissions. */
    void create_indirect_resources(const std::size_t chunk_count) {
        constexpr std::size_t spans_per_chunk = 3U + cascade_count * 2U;
        indirect_command_capacity_ = chunk_count * spans_per_chunk;
        if (indirect_command_capacity_ == 0U) {
            throw std::invalid_argument("Indirect rendering requires at least one chunk range");
        }
        const VkDeviceSize size = static_cast<VkDeviceSize>(
            indirect_command_capacity_ * sizeof(VkDrawIndexedIndirectCommand));
        create_buffer(
            size,
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            indirect_buffer_,
            indirect_memory_);
        require_success(
            vkMapMemory(device_, indirect_memory_, 0, size, 0, &indirect_mapped_),
            "Mapping the indirect draw buffer");
        indirect_commands_.reserve(indirect_command_capacity_);
    }

    /** Decodes the supplied sprite sheet and uploads it as an sRGB texture. */
    void create_atlas_resources() {
        const std::filesystem::path path =
            std::filesystem::path{VULKANCRAFT_ASSET_DIRECTORY} / "sprites" / "spritesheet.png";
        int width = 0;
        int height = 0;
        int source_channels = 0;
        stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &source_channels, STBI_rgb_alpha);
        if (pixels == nullptr || width <= 0 || height <= 0) {
            throw std::runtime_error("Could not decode block atlas: " + path.string());
        }
        if (width != 128 || height != 96) {
            stbi_image_free(pixels);
            throw std::runtime_error("The block atlas must be exactly 128 by 96 pixels");
        }

        const auto base_size = static_cast<std::size_t>(width) *
                               static_cast<std::size_t>(height) * 4U;
        const AtlasMipChain mip_chain = build_atlas_mip_chain(
            std::span<const std::uint8_t>{pixels, base_size},
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            atlas_tile_size);
        if (mip_chain.levels.size() != atlas_mip_level_count) {
            stbi_image_free(pixels);
            throw std::runtime_error("The block atlas must use 32 by 32 sprites");
        }

        const VkDeviceSize image_size = static_cast<VkDeviceSize>(mip_chain.pixels.size());
        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        try {
            create_buffer(
                image_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                staging,
                staging_memory);
            void* mapped = nullptr;
            require_success(vkMapMemory(device_, staging_memory, 0, image_size, 0, &mapped),
                            "Mapping the atlas staging buffer");
            std::memcpy(mapped, mip_chain.pixels.data(), mip_chain.pixels.size());
            vkUnmapMemory(device_, staging_memory);

            create_image(
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                1,
                VK_FORMAT_R8G8B8A8_SRGB,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                atlas_image_,
                atlas_memory_,
                atlas_mip_level_count);
            copy_buffer_to_image(staging, atlas_image_, mip_chain.levels);
            atlas_view_ = create_image_view(
                atlas_image_,
                VK_FORMAT_R8G8B8A8_SRGB,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_VIEW_TYPE_2D,
                0,
                1,
                atlas_mip_level_count);
        } catch (...) {
            stbi_image_free(pixels);
            vkDestroyBuffer(device_, staging, nullptr);
            vkFreeMemory(device_, staging_memory, nullptr);
            throw;
        }
        stbi_image_free(pixels);
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, staging_memory, nullptr);
    }

    /** Packs one replaceable mesh into a single device-local allocation. */
    [[nodiscard]] PackedMeshBufferSet upload_packed_mesh_buffer(const world::WorldMesh& mesh) {
        const bool has_indices = !mesh.indices.empty() || !mesh.transparent_indices.empty() ||
                                 !mesh.cloud_indices.empty() || !mesh.shadow_indices.empty();
        if (mesh.vertices.empty() || !has_indices) {
            throw std::invalid_argument("A packed mesh upload must not be empty");
        }
        constexpr VkDeviceSize transfer_alignment = 4U;
        const auto align_transfer = [](const VkDeviceSize value) {
            return (value + transfer_alignment - 1U) & ~(transfer_alignment - 1U);
        };
        const VkDeviceSize vertex_size = sizeof(world::Vertex) * mesh.vertices.size();
        const VkDeviceSize index_size = sizeof(std::uint32_t) * mesh.indices.size();
        const VkDeviceSize transparent_size =
            sizeof(std::uint32_t) * mesh.transparent_indices.size();
        const VkDeviceSize cloud_size = sizeof(std::uint32_t) * mesh.cloud_indices.size();
        const VkDeviceSize shadow_size = sizeof(std::uint32_t) * mesh.shadow_indices.size();

        PackedMeshBufferSet uploaded;
        uploaded.vertex_offset = 0U;
        uploaded.index_offset = align_transfer(vertex_size);
        uploaded.transparent_index_offset = align_transfer(uploaded.index_offset + index_size);
        uploaded.cloud_index_offset =
            align_transfer(uploaded.transparent_index_offset + transparent_size);
        uploaded.shadow_index_offset =
            align_transfer(uploaded.cloud_index_offset + cloud_size);
        uploaded.index_count = static_cast<std::uint32_t>(mesh.indices.size());
        uploaded.transparent_index_count =
            static_cast<std::uint32_t>(mesh.transparent_indices.size());
        uploaded.cloud_index_count = static_cast<std::uint32_t>(mesh.cloud_indices.size());
        uploaded.shadow_index_count = static_cast<std::uint32_t>(mesh.shadow_indices.size());
        const VkDeviceSize total_size = uploaded.shadow_index_offset + shadow_size;

        VkBuffer staging = VK_NULL_HANDLE;
        VkDeviceMemory staging_memory = VK_NULL_HANDLE;
        try {
            create_buffer(
                total_size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                staging,
                staging_memory);
            void* mapped = nullptr;
            require_success(
                vkMapMemory(device_, staging_memory, 0, total_size, 0, &mapped),
                "Mapping a packed mesh staging buffer");
            auto* bytes = static_cast<std::byte*>(mapped);
            std::memcpy(bytes + uploaded.vertex_offset, mesh.vertices.data(), vertex_size);
            if (index_size > 0U) {
                std::memcpy(bytes + uploaded.index_offset, mesh.indices.data(), index_size);
            }
            if (transparent_size > 0U) {
                std::memcpy(
                    bytes + uploaded.transparent_index_offset,
                    mesh.transparent_indices.data(),
                    transparent_size);
            }
            if (cloud_size > 0U) {
                std::memcpy(
                    bytes + uploaded.cloud_index_offset,
                    mesh.cloud_indices.data(),
                    cloud_size);
            }
            if (shadow_size > 0U) {
                std::memcpy(
                    bytes + uploaded.shadow_index_offset,
                    mesh.shadow_indices.data(),
                    shadow_size);
            }
            vkUnmapMemory(device_, staging_memory);
            create_buffer(
                total_size,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                uploaded.buffer,
                uploaded.memory);
            const VkCommandBuffer command = begin_one_time_commands();
            const VkBufferCopy copy{0, 0, total_size};
            vkCmdCopyBuffer(command, staging, uploaded.buffer, 1, &copy);
            VkBufferMemoryBarrier2 visibility{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            visibility.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            visibility.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            visibility.dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
            visibility.dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                       VK_ACCESS_2_INDEX_READ_BIT;
            visibility.buffer = uploaded.buffer;
            visibility.offset = 0U;
            visibility.size = total_size;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.bufferMemoryBarrierCount = 1U;
            dependency.pBufferMemoryBarriers = &visibility;
            vkCmdPipelineBarrier2(command, &dependency);
            submit_packed_upload(command, staging, staging_memory);
            staging = VK_NULL_HANDLE;
            staging_memory = VK_NULL_HANDLE;
        } catch (...) {
            vkDestroyBuffer(device_, staging, nullptr);
            vkFreeMemory(device_, staging_memory, nullptr);
            vkDestroyBuffer(device_, uploaded.buffer, nullptr);
            vkFreeMemory(device_, uploaded.memory, nullptr);
            throw;
        }
        vkDestroyBuffer(device_, staging, nullptr);
        vkFreeMemory(device_, staging_memory, nullptr);
        return uploaded;
    }

    /** Submits one packed transfer without idling the graphics queue. */
    void submit_packed_upload(
        const VkCommandBuffer command,
        const VkBuffer staging,
        const VkDeviceMemory staging_memory) {
        try {
            pending_packed_uploads_.reserve(pending_packed_uploads_.size() + 1U);
        } catch (...) {
            vkFreeCommandBuffers(device_, command_pool_, 1U, &command);
            throw;
        }
        const VkResult end_result = vkEndCommandBuffer(command);
        if (end_result != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, command_pool_, 1U, &command);
            require_success(end_result, "Ending a packed transfer command buffer");
        }
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        const VkResult fence_result = vkCreateFence(device_, &fence_info, nullptr, &fence);
        if (fence_result != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, command_pool_, 1U, &command);
            require_success(fence_result, "Creating a packed transfer fence");
        }
        VkCommandBufferSubmitInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        command_info.commandBuffer = command;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = 1U;
        submit.pCommandBufferInfos = &command_info;
        const VkResult result = vkQueueSubmit2(graphics_queue_, 1U, &submit, fence);
        if (result != VK_SUCCESS) {
            vkDestroyFence(device_, fence, nullptr);
            vkFreeCommandBuffers(device_, command_pool_, 1U, &command);
            require_success(result, "Submitting a packed transfer command buffer");
        }
        pending_packed_uploads_.push_back({
            .staging = staging,
            .staging_memory = staging_memory,
            .command = command,
            .fence = fence,
        });
    }

    /** Releases packed staging resources whose asynchronous copies are complete. */
    void release_completed_packed_uploads(const bool wait) noexcept {
        auto upload = pending_packed_uploads_.begin();
        while (upload != pending_packed_uploads_.end()) {
            if (wait) {
                static_cast<void>(vkWaitForFences(
                    device_, 1U, &upload->fence, VK_TRUE,
                    std::numeric_limits<std::uint64_t>::max()));
            } else {
                const VkResult status = vkGetFenceStatus(device_, upload->fence);
                if (status == VK_NOT_READY) {
                    ++upload;
                    continue;
                }
                if (status != VK_SUCCESS) {
                    static_cast<void>(vkWaitForFences(
                        device_, 1U, &upload->fence, VK_TRUE,
                        std::numeric_limits<std::uint64_t>::max()));
                }
            }
            vkDestroyFence(device_, upload->fence, nullptr);
            vkFreeCommandBuffers(device_, command_pool_, 1U, &upload->command);
            vkDestroyBuffer(device_, upload->staging, nullptr);
            vkFreeMemory(device_, upload->staging_memory, nullptr);
            upload = pending_packed_uploads_.erase(upload);
        }
    }

    /** Uploads several independent meshes with one graphics-queue submission. */
    [[nodiscard]] std::vector<MeshBufferSet> upload_mesh_buffer_batch(
        const std::span<const world::WorldMesh* const> meshes) {
        const auto prepare_upload = [this](
                                        const void* data,
                                        const VkDeviceSize size,
                                        const VkBufferUsageFlags usage,
                                        VkBuffer& staging,
                                        VkDeviceMemory& staging_memory,
                                        VkBuffer& destination,
                                        VkDeviceMemory& destination_memory) {
            create_buffer(
                size,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                staging,
                staging_memory);
            void* mapped = nullptr;
            require_success(
                vkMapMemory(device_, staging_memory, 0, size, 0, &mapped),
                "Mapping a mesh staging buffer");
            std::memcpy(mapped, data, static_cast<std::size_t>(size));
            vkUnmapMemory(device_, staging_memory);
            create_buffer(
                size,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                destination,
                destination_memory);
        };
        const auto release_staging = [this](PendingMeshUpload& upload) {
            vkDestroyBuffer(device_, upload.shadow_staging, nullptr);
            vkFreeMemory(device_, upload.shadow_staging_memory, nullptr);
            vkDestroyBuffer(device_, upload.transparent_staging, nullptr);
            vkFreeMemory(device_, upload.transparent_staging_memory, nullptr);
            vkDestroyBuffer(device_, upload.cloud_staging, nullptr);
            vkFreeMemory(device_, upload.cloud_staging_memory, nullptr);
            vkDestroyBuffer(device_, upload.index_staging, nullptr);
            vkFreeMemory(device_, upload.index_staging_memory, nullptr);
            vkDestroyBuffer(device_, upload.vertex_staging, nullptr);
            vkFreeMemory(device_, upload.vertex_staging_memory, nullptr);
            upload.shadow_staging = VK_NULL_HANDLE;
            upload.shadow_staging_memory = VK_NULL_HANDLE;
            upload.transparent_staging = VK_NULL_HANDLE;
            upload.transparent_staging_memory = VK_NULL_HANDLE;
            upload.cloud_staging = VK_NULL_HANDLE;
            upload.cloud_staging_memory = VK_NULL_HANDLE;
            upload.index_staging = VK_NULL_HANDLE;
            upload.index_staging_memory = VK_NULL_HANDLE;
            upload.vertex_staging = VK_NULL_HANDLE;
            upload.vertex_staging_memory = VK_NULL_HANDLE;
        };

        std::vector<PendingMeshUpload> pending;
        pending.reserve(meshes.size());
        try {
            for (const world::WorldMesh* mesh : meshes) {
                if (mesh == nullptr || mesh->vertices.empty() || mesh->indices.empty()) {
                    throw std::invalid_argument("A batched mesh upload must not contain empty meshes");
                }
                pending.emplace_back();
                PendingMeshUpload& upload = pending.back();
                upload.vertex_size = sizeof(world::Vertex) * mesh->vertices.size();
                upload.index_size = sizeof(std::uint32_t) * mesh->indices.size();
                upload.transparent_index_size =
                    sizeof(std::uint32_t) * mesh->transparent_indices.size();
                upload.cloud_index_size = sizeof(std::uint32_t) * mesh->cloud_indices.size();
                upload.shadow_index_size =
                    sizeof(std::uint32_t) * mesh->shadow_indices.size();
                prepare_upload(
                    mesh->vertices.data(),
                    upload.vertex_size,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    upload.vertex_staging,
                    upload.vertex_staging_memory,
                    upload.destination.vertex_buffer,
                    upload.destination.vertex_memory);
                prepare_upload(
                    mesh->indices.data(),
                    upload.index_size,
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    upload.index_staging,
                    upload.index_staging_memory,
                    upload.destination.index_buffer,
                    upload.destination.index_memory);
                if (!mesh->transparent_indices.empty()) {
                    prepare_upload(
                        mesh->transparent_indices.data(),
                        upload.transparent_index_size,
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        upload.transparent_staging,
                        upload.transparent_staging_memory,
                        upload.destination.transparent_index_buffer,
                        upload.destination.transparent_index_memory);
                }
                if (!mesh->cloud_indices.empty()) {
                    prepare_upload(
                        mesh->cloud_indices.data(),
                        upload.cloud_index_size,
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        upload.cloud_staging,
                        upload.cloud_staging_memory,
                        upload.destination.cloud_index_buffer,
                        upload.destination.cloud_index_memory);
                }
                if (!mesh->shadow_indices.empty()) {
                    prepare_upload(
                        mesh->shadow_indices.data(),
                        upload.shadow_index_size,
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        upload.shadow_staging,
                        upload.shadow_staging_memory,
                        upload.destination.shadow_index_buffer,
                        upload.destination.shadow_index_memory);
                }
                upload.destination.index_count = static_cast<std::uint32_t>(mesh->indices.size());
                upload.destination.transparent_index_count =
                    static_cast<std::uint32_t>(mesh->transparent_indices.size());
                upload.destination.cloud_index_count =
                    static_cast<std::uint32_t>(mesh->cloud_indices.size());
                upload.destination.shadow_index_count =
                    static_cast<std::uint32_t>(mesh->shadow_indices.size());
            }

            if (!pending.empty()) {
                const VkCommandBuffer command = begin_one_time_commands();
                for (const PendingMeshUpload& upload : pending) {
                    const VkBufferCopy vertex_copy{0, 0, upload.vertex_size};
                    const VkBufferCopy index_copy{0, 0, upload.index_size};
                    vkCmdCopyBuffer(
                        command,
                        upload.vertex_staging,
                        upload.destination.vertex_buffer,
                        1,
                        &vertex_copy);
                    vkCmdCopyBuffer(
                        command,
                        upload.index_staging,
                        upload.destination.index_buffer,
                        1,
                        &index_copy);
                    if (upload.transparent_staging != VK_NULL_HANDLE) {
                        const VkBufferCopy transparent_copy{0, 0, upload.transparent_index_size};
                        vkCmdCopyBuffer(
                            command,
                            upload.transparent_staging,
                            upload.destination.transparent_index_buffer,
                            1,
                            &transparent_copy);
                    }
                    if (upload.cloud_staging != VK_NULL_HANDLE) {
                        const VkBufferCopy cloud_copy{0, 0, upload.cloud_index_size};
                        vkCmdCopyBuffer(
                            command,
                            upload.cloud_staging,
                            upload.destination.cloud_index_buffer,
                            1,
                            &cloud_copy);
                    }
                    if (upload.shadow_staging != VK_NULL_HANDLE) {
                        const VkBufferCopy shadow_copy{0, 0, upload.shadow_index_size};
                        vkCmdCopyBuffer(
                            command,
                            upload.shadow_staging,
                            upload.destination.shadow_index_buffer,
                            1,
                            &shadow_copy);
                    }
                }
                end_one_time_commands(command);
            }
        } catch (...) {
            for (PendingMeshUpload& upload : pending) {
                release_staging(upload);
                destroy_mesh_buffer_set(upload.destination);
            }
            throw;
        }

        std::vector<MeshBufferSet> uploaded;
        uploaded.reserve(pending.size());
        for (PendingMeshUpload& upload : pending) {
            release_staging(upload);
            uploaded.push_back(upload.destination);
            upload.destination = {};
        }
        return uploaded;
    }

    /** Uploads one immutable mesh through the shared transfer path. */
    [[nodiscard]] MeshBufferSet upload_mesh_buffers(const world::WorldMesh& mesh) {
        const std::array<const world::WorldMesh*, 1> meshes{&mesh};
        auto uploaded = upload_mesh_buffer_batch(meshes);
        return uploaded.front();
    }

    /** Uploads and installs a complete streamed-world mesh. */
    void create_mesh_buffers(const world::WorldMesh& mesh) {
        MeshBufferSet uploaded = upload_mesh_buffers(mesh);
        vertex_buffer_ = uploaded.vertex_buffer;
        vertex_memory_ = uploaded.vertex_memory;
        index_buffer_ = uploaded.index_buffer;
        index_memory_ = uploaded.index_memory;
        index_count_ = uploaded.index_count;
        transparent_index_buffer_ = uploaded.transparent_index_buffer;
        transparent_index_memory_ = uploaded.transparent_index_memory;
        transparent_index_count_ = uploaded.transparent_index_count;
        cloud_index_buffer_ = uploaded.cloud_index_buffer;
        cloud_index_memory_ = uploaded.cloud_index_memory;
        cloud_index_count_ = uploaded.cloud_index_count;
        shadow_index_buffer_ = uploaded.shadow_index_buffer;
        shadow_index_memory_ = uploaded.shadow_index_memory;
        shadow_index_count_ = uploaded.shadow_index_count;
        draw_ranges_ = mesh.draw_ranges;
        if (draw_ranges_.empty()) {
            draw_ranges_.push_back({
                .first_index = 0,
                .index_count = index_count_,
                .first_transparent_index = 0,
                .transparent_index_count = transparent_index_count_,
                .first_cloud_index = 0,
                .cloud_index_count = cloud_index_count_,
                .first_shadow_index = 0,
                .shadow_index_count = shadow_index_count_,
            });
        }
        chunk_overrides_.resize(draw_ranges_.size());
        base_range_for_slot_.resize(draw_ranges_.size());
        std::iota(base_range_for_slot_.begin(), base_range_for_slot_.end(), 0U);
    }

    /** Destroys every handle owned by one independently uploaded mesh. */
    void destroy_mesh_buffer_set(MeshBufferSet& buffers) noexcept {
        vkDestroyBuffer(device_, buffers.shadow_index_buffer, nullptr);
        vkFreeMemory(device_, buffers.shadow_index_memory, nullptr);
        vkDestroyBuffer(device_, buffers.transparent_index_buffer, nullptr);
        vkFreeMemory(device_, buffers.transparent_index_memory, nullptr);
        vkDestroyBuffer(device_, buffers.cloud_index_buffer, nullptr);
        vkFreeMemory(device_, buffers.cloud_index_memory, nullptr);
        vkDestroyBuffer(device_, buffers.index_buffer, nullptr);
        vkFreeMemory(device_, buffers.index_memory, nullptr);
        vkDestroyBuffer(device_, buffers.vertex_buffer, nullptr);
        vkFreeMemory(device_, buffers.vertex_memory, nullptr);
        buffers = {};
    }

    /** Releases all renderer-side chunk replacements. */
    void destroy_chunk_overrides() noexcept {
        chunk_overrides_.clear();
    }

    /** Allocates the layered depth image and views used by four shadow passes. */
    void create_shadow_resources() {
        shadow_format_ = find_depth_format(true);
        create_image(
            shadow_resolution,
            shadow_resolution,
            static_cast<std::uint32_t>(cascade_count),
            shadow_format_,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            shadow_image_,
            shadow_memory_);
        shadow_array_view_ = create_image_view(
            shadow_image_,
            shadow_format_,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            0,
            static_cast<std::uint32_t>(cascade_count));
        for (std::uint32_t layer = 0; layer < static_cast<std::uint32_t>(cascade_count); ++layer) {
            shadow_layer_views_[layer] = create_image_view(
                shadow_image_, shadow_format_, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_VIEW_TYPE_2D, layer, 1);
        }
    }

    /** Allocates the swapchain-sized scene depth target. */
    void create_depth_resources() {
        depth_format_ = find_depth_format(false);
        create_image(
            swapchain_extent_.width,
            swapchain_extent_.height,
            1,
            depth_format_,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            depth_image_,
            depth_memory_);
        depth_view_ = create_image_view(
            depth_image_, depth_format_, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_VIEW_TYPE_2D, 0, 1);
        depth_initialized_ = false;
    }

    /** Allocates the HDR scene target consumed by the post-processing pipeline. */
    void create_scene_color_resources() {
        create_image(
            swapchain_extent_.width,
            swapchain_extent_.height,
            1,
            scene_color_format,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            scene_color_image_,
            scene_color_memory_);
        scene_color_view_ = create_image_view(
            scene_color_image_,
            scene_color_format,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_VIEW_TYPE_2D,
            0,
            1);
        scene_color_initialized_ = false;
    }

    /** Updates the single descriptor set after its shadow image exists. */
    void update_descriptor_set() {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = uniform_buffer_;
        buffer_info.range = sizeof(FrameUniforms);

        VkDescriptorImageInfo image_info{};
        image_info.sampler = shadow_sampler_;
        image_info.imageView = shadow_array_view_;
        image_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo atlas_info{};
        atlas_info.sampler = atlas_sampler_;
        atlas_info.imageView = atlas_view_;
        atlas_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo scene_info{};
        scene_info.sampler = scene_sampler_;
        scene_info.imageView = scene_color_view_;
        scene_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo depth_info{};
        depth_info.sampler = depth_sampler_;
        depth_info.imageView = depth_view_;
        depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 5> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptor_set_;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo = &buffer_info;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptor_set_;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &image_info;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptor_set_;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].pImageInfo = &atlas_info;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = descriptor_set_;
        writes[3].dstBinding = 3;
        writes[3].descriptorCount = 1;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].pImageInfo = &scene_info;
        writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4].dstSet = descriptor_set_;
        writes[4].dstBinding = 4;
        writes[4].descriptorCount = 1;
        writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4].pImageInfo = &depth_info;
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    /** Creates descriptor-compatible layouts and both graphics pipelines. */
    void create_pipelines() {
        VkPushConstantRange celestial_range{};
        celestial_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        celestial_range.size = sizeof(CelestialPushConstants);
        VkPipelineLayoutCreateInfo world_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        world_layout_info.setLayoutCount = 1;
        world_layout_info.pSetLayouts = &descriptor_set_layout_;
        world_layout_info.pushConstantRangeCount = 1;
        world_layout_info.pPushConstantRanges = &celestial_range;
        require_success(vkCreatePipelineLayout(device_, &world_layout_info, nullptr, &world_pipeline_layout_),
                        "Creating the world pipeline layout");

        VkPushConstantRange cascade_range{};
        cascade_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        cascade_range.size = sizeof(std::int32_t);
        VkPipelineLayoutCreateInfo shadow_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        shadow_layout_info.setLayoutCount = 1;
        shadow_layout_info.pSetLayouts = &descriptor_set_layout_;
        shadow_layout_info.pushConstantRangeCount = 1;
        shadow_layout_info.pPushConstantRanges = &cascade_range;
        require_success(vkCreatePipelineLayout(device_, &shadow_layout_info, nullptr, &shadow_pipeline_layout_),
                        "Creating the shadow pipeline layout");

        world_pipeline_ = create_graphics_pipeline(PipelineKind::opaque);
        wireframe_pipeline_ = create_graphics_pipeline(PipelineKind::wireframe);
        transparent_pipeline_ = create_graphics_pipeline(PipelineKind::transparent);
        cloud_pipeline_ = create_graphics_pipeline(PipelineKind::cloud);
        depth_prepass_pipeline_ = create_graphics_pipeline(PipelineKind::depth_prepass);
        cloud_depth_prepass_pipeline_ =
            create_graphics_pipeline(PipelineKind::cloud_depth_prepass);
        shadow_pipeline_ = create_graphics_pipeline(PipelineKind::shadow);
        sky_pipeline_ = create_graphics_pipeline(PipelineKind::sky);
        celestial_pipeline_ = create_graphics_pipeline(PipelineKind::celestial);
        postprocess_pipeline_ = create_graphics_pipeline(PipelineKind::postprocess);
    }

    /** Builds either the color pipeline or the depth-only cascade pipeline. */
    [[nodiscard]] VkPipeline create_graphics_pipeline(const PipelineKind kind) const {
        const bool shadow = kind == PipelineKind::shadow;
        const bool depth_prepass = kind == PipelineKind::depth_prepass;
        const bool cloud_depth_prepass = kind == PipelineKind::cloud_depth_prepass;
        const bool scene_depth_prepass = depth_prepass || cloud_depth_prepass;
        const bool sky = kind == PipelineKind::sky;
        const bool celestial = kind == PipelineKind::celestial;
        const bool postprocess = kind == PipelineKind::postprocess;
        const bool wireframe = kind == PipelineKind::wireframe;
        const bool cloud = kind == PipelineKind::cloud;
        const bool transparent = kind == PipelineKind::transparent || cloud;
        const bool color_pass = !shadow && !scene_depth_prepass;
        const bool fragment_stage = color_pass || scene_depth_prepass || shadow;
        const std::filesystem::path shader_directory{VULKANCRAFT_SHADER_DIRECTORY};
        const char* vertex_name = shadow ? "shadow.vert.spv"
                                   : scene_depth_prepass ? "depth_prepass.vert.spv"
                                   : sky ? "sky.vert.spv"
                                   : celestial ? "celestial.vert.spv"
                                   : postprocess ? "postprocess.vert.spv"
                                         : "world.vert.spv";
        const auto vertex_words = read_spirv(shader_directory / vertex_name);
        const VkShaderModule vertex_module = create_shader_module(vertex_words);
        VkShaderModule fragment_module = VK_NULL_HANDLE;

        try {
            std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertex_module;
            stages[0].pName = "main";
            std::uint32_t stage_count = 1;
            if (fragment_stage) {
                fragment_module = create_shader_module(read_spirv(
                    shader_directory /
                    (shadow ? "shadow.frag.spv"
                      : scene_depth_prepass ? "depth_prepass.frag.spv"
                      : sky ? "sky.frag.spv"
                      : celestial ? "celestial.frag.spv"
                      : postprocess ? "postprocess.frag.spv"
                                  : "world.frag.spv")));
                stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                stages[1].module = fragment_module;
                stages[1].pName = "main";
                stage_count = 2;
            }

            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(world::Vertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            const std::array<VkVertexInputAttributeDescription, 3> attributes{{
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(world::Vertex, position)},
                {1, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(world::Vertex, texture_coordinate)},
                {2, 0, VK_FORMAT_R32_UINT, offsetof(world::Vertex, attributes)},
            }};
            const std::array<VkVertexInputAttributeDescription, 3> depth_attributes{{
                attributes[0], attributes[1], attributes[2]}};
            const std::array<VkVertexInputAttributeDescription, 3> shadow_attributes{{
                attributes[0], attributes[1], attributes[2]}};
            VkPipelineVertexInputStateCreateInfo vertex_input{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            const bool generated_vertices = sky || celestial || postprocess;
            vertex_input.vertexBindingDescriptionCount = generated_vertices ? 0U : 1U;
            vertex_input.pVertexBindingDescriptions = generated_vertices ? nullptr : &binding;
            vertex_input.vertexAttributeDescriptionCount = generated_vertices ? 0U
                : (shadow ? static_cast<std::uint32_t>(shadow_attributes.size())
                          : scene_depth_prepass
                                ? static_cast<std::uint32_t>(depth_attributes.size())
                                : static_cast<std::uint32_t>(attributes.size()));
            vertex_input.pVertexAttributeDescriptions = generated_vertices ? nullptr
                : (shadow ? shadow_attributes.data()
                          : scene_depth_prepass ? depth_attributes.data()
                                                : attributes.data());

            VkPipelineInputAssemblyStateCreateInfo input_assembly{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterization{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rasterization.polygonMode = wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
            rasterization.cullMode = wireframe || transparent || cloud_depth_prepass || sky ||
                                             celestial || postprocess
                                         ? VK_CULL_MODE_NONE
                                         : VK_CULL_MODE_BACK_BIT;
            rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterization.depthBiasEnable = shadow ? VK_TRUE : VK_FALSE;
            rasterization.lineWidth = 1.0F;

            VkPipelineMultisampleStateCreateInfo multisample{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth_stencil{
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth_stencil.depthTestEnable = sky || postprocess ? VK_FALSE : VK_TRUE;
            depth_stencil.depthWriteEnable = shadow || scene_depth_prepass || wireframe ||
                                                     kind == PipelineKind::transparent
                                                 ? VK_TRUE
                                                 : VK_FALSE;
            depth_stencil.depthCompareOp = kind == PipelineKind::opaque || cloud
                                                ? VK_COMPARE_OP_EQUAL
                                                : (wireframe || celestial
                                                       ? VK_COMPARE_OP_LESS_OR_EQUAL
                                                       : VK_COMPARE_OP_LESS);

            VkPipelineColorBlendAttachmentState blend_attachment{};
            blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            if (transparent) {
                blend_attachment.blendEnable = VK_TRUE;
                blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
                blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
            }
            VkPipelineColorBlendStateCreateInfo blend{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blend.attachmentCount = color_pass ? 1U : 0U;
            blend.pAttachments = color_pass ? &blend_attachment : nullptr;

            constexpr std::array dynamic_states{
                VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
            VkPipelineDynamicStateCreateInfo dynamic{
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount = shadow ? static_cast<std::uint32_t>(dynamic_states.size()) : 2U;
            dynamic.pDynamicStates = dynamic_states.data();

            VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            const VkFormat color_format = postprocess ? swapchain_format_ : scene_color_format;
            rendering.colorAttachmentCount = color_pass ? 1U : 0U;
            rendering.pColorAttachmentFormats = color_pass ? &color_format : nullptr;
            rendering.depthAttachmentFormat = shadow ? shadow_format_
                                              : postprocess ? VK_FORMAT_UNDEFINED
                                                            : depth_format_;

            VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeline_info.pNext = &rendering;
            pipeline_info.stageCount = stage_count;
            pipeline_info.pStages = stages.data();
            pipeline_info.pVertexInputState = &vertex_input;
            pipeline_info.pInputAssemblyState = &input_assembly;
            pipeline_info.pViewportState = &viewport_state;
            pipeline_info.pRasterizationState = &rasterization;
            pipeline_info.pMultisampleState = &multisample;
            pipeline_info.pDepthStencilState = &depth_stencil;
            pipeline_info.pColorBlendState = &blend;
            pipeline_info.pDynamicState = &dynamic;
            pipeline_info.layout = shadow ? shadow_pipeline_layout_ : world_pipeline_layout_;

            VkPipeline pipeline = VK_NULL_HANDLE;
            require_success(vkCreateGraphicsPipelines(
                                device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline),
                            shadow ? "Creating the shadow pipeline" : "Creating the world pipeline");
            vkDestroyShaderModule(device_, vertex_module, nullptr);
            if (fragment_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, fragment_module, nullptr);
            }
            return pipeline;
        } catch (...) {
            vkDestroyShaderModule(device_, vertex_module, nullptr);
            if (fragment_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device_, fragment_module, nullptr);
            }
            throw;
        }
    }

    /** Creates a Vulkan module from aligned SPIR-V words. */
    [[nodiscard]] VkShaderModule create_shader_module(
        const std::vector<std::uint32_t>& words) const {
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = words.size() * sizeof(std::uint32_t);
        info.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        require_success(vkCreateShaderModule(device_, &info, nullptr, &module), "Creating a shader module");
        return module;
    }

    /** Allocates the persistent primary frame command buffer. */
    void allocate_command_buffer() {
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = command_pool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        require_success(vkAllocateCommandBuffers(device_, &info, &command_buffer_),
                        "Allocating the frame command buffer");
    }

    /** Creates the binary semaphores and one-frame-in-flight fence. */
    void create_synchronization() {
        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        require_success(vkCreateSemaphore(device_, &semaphore_info, nullptr, &image_available_),
                        "Creating the image-available semaphore");
        create_render_finished_semaphores();
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        require_success(vkCreateFence(device_, &fence_info, nullptr, &frame_fence_),
                        "Creating the frame fence");
    }

    /** Creates one presentation semaphore per swapchain image for safe reuse. */
    void create_render_finished_semaphores() {
        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        render_finished_.resize(swapchain_images_.size(), VK_NULL_HANDLE);
        for (VkSemaphore& semaphore : render_finished_) {
            require_success(vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore),
                            "Creating a render-finished semaphore");
        }
    }

    /** Recalculates camera and animated directional-light frame data. */
    void update_uniforms(
        const core::Camera& camera,
        const float world_time_seconds,
        const float animation_seconds,
        const bool underwater) {
        const float aspect = static_cast<float>(swapchain_extent_.width) /
                             static_cast<float>(swapchain_extent_.height);
        const float day_angle = celestial_angle(world_time_seconds);
        const glm::vec3 sun_to_source = celestial_direction(day_angle);
        const bool daytime = sun_to_source.y >= 0.0F;
        const glm::vec3 active_light_to_source = daytime ? sun_to_source : -sun_to_source;
        const glm::vec3 light_direction = -active_light_to_source;
        const float daylight = std::clamp(sun_to_source.y * 3.0F + 0.22F, 0.0F, 1.0F);
        const float light_intensity = daytime ? std::lerp(0.38F, 0.86F, daylight) : 0.30F;
        const glm::vec3 light_color = daytime
                                          ? glm::mix(
                                                glm::vec3{1.0F, 0.72F, 0.54F},
                                                glm::vec3{1.0F, 0.96F, 0.84F},
                                                daylight)
                                          : glm::vec3{0.48F, 0.58F, 0.86F};
        current_sky_color_ = glm::mix(
            glm::vec3{0.018F, 0.028F, 0.075F},
            glm::vec3{0.20F, 0.46F, 0.78F},
            daylight);
        const auto calculated_cascades = cascade_calculator_.calculate(
            camera, aspect, light_direction, shadow_resolution);
        for (std::size_t cascade = 0; cascade < cascade_count; ++cascade) {
            const std::uint64_t update_interval = 1ULL << cascade;
            shadow_cascade_updates_[cascade] =
                !shadow_initialized_ || frame_sequence_ % update_interval == 0U;
            if (shadow_cascade_updates_[cascade]) {
                cached_shadow_matrices_[cascade] =
                    calculated_cascades.light_view_projections[cascade];
            }
        }

        FrameUniforms uniforms;
        uniforms.view = camera.view_matrix();
        uniforms.projection = camera.projection_matrix(aspect);
        uniforms.light_view_projection = cached_shadow_matrices_;
        uniforms.cascade_splits = glm::vec4{
            calculated_cascades.split_depths[0],
            calculated_cascades.split_depths[1],
            calculated_cascades.split_depths[2],
            calculated_cascades.split_depths[3],
        };
        uniforms.light_direction = glm::vec4(light_direction, 0.0F);
        uniforms.camera_position = glm::vec4(camera.position(), underwater ? 1.0F : 0.0F);
        uniforms.light_color_intensity = glm::vec4(light_color, light_intensity);
        uniforms.sky_color = glm::vec4(current_sky_color_, daylight);
        uniforms.sun_direction = glm::vec4(sun_to_source, 0.0F);
        uniforms.animation_data = glm::vec4(animation_seconds, 0.0F, 0.0F, 0.0F);
        uniforms.console_state = glm::uvec4(
            console_visible_ ? 1U : 0U,
            static_cast<std::uint32_t>(console_text_.size()),
            0U,
            0U);
        for (std::size_t index = 0; index < console_text_.size(); ++index) {
            uniforms.console_text[index / 4U][index % 4U] =
                static_cast<std::uint32_t>(static_cast<unsigned char>(console_text_[index]));
        }
        update_chunk_visibility(
            uniforms.projection * uniforms.view,
            cached_shadow_matrices_,
            calculated_cascades.split_depths,
            camera.position(),
            animation_seconds);
        current_sun_direction_ = sun_to_source;
        current_underwater_ = underwater;
        std::memcpy(uniform_mapped_, &uniforms, sizeof(uniforms));
    }

    /** Builds conservative per-pass chunk lists before command recording. */
    void update_chunk_visibility(
        const glm::mat4& scene_clip_from_world,
        const std::array<glm::mat4, cascade_count>& shadow_clip_from_world,
        const std::array<float, cascade_count>& shadow_split_depths,
        const glm::vec3& camera_position,
        const float elapsed_seconds) {
        const Frustum scene_frustum = Frustum::from_clip_matrix(scene_clip_from_world);
        std::array<Frustum, cascade_count> shadow_frusta{
            Frustum::from_clip_matrix(shadow_clip_from_world[0]),
            Frustum::from_clip_matrix(shadow_clip_from_world[1]),
            Frustum::from_clip_matrix(shadow_clip_from_world[2]),
            Frustum::from_clip_matrix(shadow_clip_from_world[3]),
        };
        scene_visible_slots_.clear();
        for (auto& slots : shadow_visible_slots_) {
            slots.clear();
        }
        if (active_chunk_coordinates_.size() != draw_ranges_.size()) {
            throw std::logic_error("Chunk coordinates and draw ranges are inconsistent");
        }
        scene_visible_slots_.reserve(active_chunk_coordinates_.size());
        for (auto& slots : shadow_visible_slots_) {
            slots.reserve(active_chunk_coordinates_.size());
        }
        for (std::size_t slot = 0; slot < active_chunk_coordinates_.size(); ++slot) {
            const world::ChunkCoordinate coordinate = active_chunk_coordinates_[slot];
            const AxisAlignedBoundingBox bounds = chunk_bounds(coordinate, elapsed_seconds);
            if (scene_frustum.intersects(bounds)) {
                scene_visible_slots_.push_back(slot);
            }
            constexpr float caster_guard_distance = 96.0F;
            const glm::vec2 chunk_center{
                (bounds.minimum.x + bounds.maximum.x) * 0.5F,
                (bounds.minimum.z + bounds.maximum.z) * 0.5F};
            const float horizontal_distance = glm::distance(
                chunk_center, glm::vec2{camera_position.x, camera_position.z});
            for (std::size_t cascade = 0; cascade < cascade_count; ++cascade) {
                if (horizontal_distance <=
                        shadow_split_depths[cascade] + caster_guard_distance &&
                    shadow_frusta[cascade].intersects(bounds)) {
                    shadow_visible_slots_[cascade].push_back(slot);
                }
            }
        }
        update_indirect_commands();
    }

    /** Rebuilds compact multi-draw lists for the chunks accepted by CPU frustum culling. */
    void update_indirect_commands() {
        indirect_commands_.clear();
        const auto append_span = [this](
                                     IndirectDrawSpan& span,
                                     const std::span<const std::size_t> slots,
                                     const auto index_count,
                                     const auto first_index) {
            span.offset = static_cast<VkDeviceSize>(
                indirect_commands_.size() * sizeof(VkDrawIndexedIndirectCommand));
            for (const std::size_t slot : slots) {
                if (chunk_overrides_[slot].has_value()) {
                    continue;
                }
                const world::MeshDrawRange& range = draw_ranges_[base_range_for_slot_[slot]];
                const std::uint32_t count = index_count(range);
                if (count == 0U) {
                    continue;
                }
                indirect_commands_.push_back({
                    .indexCount = count,
                    .instanceCount = 1U,
                    .firstIndex = first_index(range),
                    .vertexOffset = 0,
                    .firstInstance = 0U,
                });
            }
            span.count = static_cast<std::uint32_t>(
                indirect_commands_.size() - span.offset / sizeof(VkDrawIndexedIndirectCommand));
        };

        append_span(
            opaque_indirect_,
            scene_visible_slots_,
            [](const world::MeshDrawRange& range) { return range.index_count; },
            [](const world::MeshDrawRange& range) { return range.first_index; });
        append_span(
            transparent_indirect_,
            scene_visible_slots_,
            [](const world::MeshDrawRange& range) { return range.transparent_index_count; },
            [](const world::MeshDrawRange& range) { return range.first_transparent_index; });
        append_span(
            cloud_indirect_,
            scene_visible_slots_,
            [](const world::MeshDrawRange& range) { return range.cloud_index_count; },
            [](const world::MeshDrawRange& range) { return range.first_cloud_index; });
        for (std::size_t cascade = 0; cascade < cascade_count; ++cascade) {
            append_span(
                shadow_opaque_indirect_[cascade],
                shadow_visible_slots_[cascade],
                [](const world::MeshDrawRange& range) { return range.index_count; },
                [](const world::MeshDrawRange& range) { return range.first_index; });
            append_span(
                shadow_cloud_indirect_[cascade],
                shadow_visible_slots_[cascade],
                [](const world::MeshDrawRange& range) { return range.shadow_index_count; },
                [](const world::MeshDrawRange& range) { return range.first_shadow_index; });
        }
        if (indirect_commands_.size() > indirect_command_capacity_) {
            throw std::runtime_error("The indirect draw buffer capacity was exceeded");
        }
        std::memcpy(
            indirect_mapped_,
            indirect_commands_.data(),
            indirect_commands_.size() * sizeof(VkDrawIndexedIndirectCommand));
    }

    /** Records four depth cascades followed by the lit world color pass. */
    void record_commands(const std::uint32_t image_index) {
        require_success(vkResetCommandBuffer(command_buffer_, 0), "Resetting the frame command buffer");
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        require_success(vkBeginCommandBuffer(command_buffer_, &begin_info), "Beginning the frame command buffer");

        transition_shadow_for_writes();
        record_shadow_passes();
        transition_shadow_for_reads();
        transition_scene_attachments();
        record_depth_prepass();
        record_world_pass();
        if (!wireframe_enabled_) {
            record_cloud_depth_prepass();
            record_transparent_pass();
        }
        transition_scene_for_postprocess();
        transition_swapchain_for_postprocess(image_index);
        record_postprocess_pass(image_index);
        transition_color_for_present(image_index);

        require_success(vkEndCommandBuffer(command_buffer_), "Ending the frame command buffer");
        shadow_initialized_ = true;
        depth_initialized_ = true;
        scene_color_initialized_ = true;
        swapchain_image_initialized_[image_index] = true;
    }

    /** Returns true when at least one original chunk batch has been replaced. */
    [[nodiscard]] bool has_chunk_overrides() const noexcept {
        return std::any_of(
            chunk_overrides_.begin(),
            chunk_overrides_.end(),
            [](const auto& mesh) { return mesh.has_value(); });
    }

    /** Returns true when the original aggregate cannot be submitted as one draw. */
    [[nodiscard]] bool uses_chunk_remapping() const noexcept {
        if (has_chunk_overrides() || base_range_for_slot_.size() != draw_ranges_.size()) {
            return true;
        }
        for (std::size_t index = 0; index < base_range_for_slot_.size(); ++index) {
            if (base_range_for_slot_[index] != index) {
                return true;
            }
        }
        return false;
    }

    /** Draws opaque chunk batches, substituting edited chunks when present. */
    void draw_opaque_geometry(const std::span<const std::size_t> visible_slots) {
        const VkDeviceSize offset = 0;
        if (!uses_chunk_remapping() && visible_slots.size() == draw_ranges_.size()) {
            vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
            vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(command_buffer_, index_count_, 1, 0, 0, 0);
            return;
        }
        if (!has_chunk_overrides()) {
            vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
            vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexedIndirect(
                command_buffer_,
                indirect_buffer_,
                opaque_indirect_.offset,
                opaque_indirect_.count,
                sizeof(VkDrawIndexedIndirectCommand));
            return;
        }
        for (const std::size_t index : visible_slots) {
            if (chunk_overrides_[index].has_value()) {
                const ChunkMeshBinding& binding = *chunk_overrides_[index];
                const PackedMeshBufferSet& buffers = binding.owner->buffers;
                vkCmdBindVertexBuffers(
                    command_buffer_, 0, 1, &buffers.buffer, &buffers.vertex_offset);
                vkCmdBindIndexBuffer(
                    command_buffer_, buffers.buffer, buffers.index_offset, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(
                    command_buffer_, binding.range.index_count, 1, binding.range.first_index, 0, 0);
            } else {
                const world::MeshDrawRange& range =
                    draw_ranges_[base_range_for_slot_[index]];
                vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
                vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(
                    command_buffer_, range.index_count, 1, range.first_index, 0, 0);
            }
        }
    }

    /** Draws either cloud or water batches using their required pass ordering. */
    void draw_transparent_geometry(const bool draw_clouds) {
        const VkDeviceSize offset = 0;
        const VkBuffer base_index_buffer =
            draw_clouds ? cloud_index_buffer_ : transparent_index_buffer_;
        const std::uint32_t base_index_count =
            draw_clouds ? cloud_index_count_ : transparent_index_count_;
        if (!uses_chunk_remapping() && scene_visible_slots_.size() == draw_ranges_.size()) {
            if (base_index_count > 0U) {
                vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
                vkCmdBindIndexBuffer(
                    command_buffer_, base_index_buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(command_buffer_, base_index_count, 1, 0, 0, 0);
            }
            return;
        }
        if (!has_chunk_overrides()) {
            const IndirectDrawSpan& span = draw_clouds ? cloud_indirect_ : transparent_indirect_;
            if (span.count > 0U) {
                vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
                vkCmdBindIndexBuffer(
                    command_buffer_, base_index_buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexedIndirect(
                    command_buffer_,
                    indirect_buffer_,
                    span.offset,
                    span.count,
                    sizeof(VkDrawIndexedIndirectCommand));
            }
            return;
        }
        for (const std::size_t index : scene_visible_slots_) {
            if (chunk_overrides_[index].has_value()) {
                const ChunkMeshBinding& binding = *chunk_overrides_[index];
                const PackedMeshBufferSet& buffers = binding.owner->buffers;
                const std::uint32_t index_count = draw_clouds
                                                      ? binding.range.cloud_index_count
                                                      : binding.range.transparent_index_count;
                const std::uint32_t first_index = draw_clouds
                                                      ? binding.range.first_cloud_index
                                                      : binding.range.first_transparent_index;
                const VkDeviceSize index_offset = draw_clouds
                                                      ? buffers.cloud_index_offset
                                                      : buffers.transparent_index_offset;
                if (index_count == 0U) {
                    continue;
                }
                vkCmdBindVertexBuffers(
                    command_buffer_, 0, 1, &buffers.buffer, &buffers.vertex_offset);
                vkCmdBindIndexBuffer(
                    command_buffer_,
                    buffers.buffer,
                    index_offset,
                    VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(
                    command_buffer_,
                    index_count,
                    1,
                    first_index,
                    0,
                    0);
            } else {
                const world::MeshDrawRange& range =
                    draw_ranges_[base_range_for_slot_[index]];
                const std::uint32_t index_count =
                    draw_clouds ? range.cloud_index_count : range.transparent_index_count;
                const std::uint32_t first_index =
                    draw_clouds ? range.first_cloud_index : range.first_transparent_index;
                if (index_count == 0U) {
                    continue;
                }
                vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
                vkCmdBindIndexBuffer(
                    command_buffer_, base_index_buffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(
                    command_buffer_,
                    index_count,
                    1,
                    first_index,
                    0,
                    0);
            }
        }
    }

    /** Draws opaque terrain plus cloud-only caster batches into one cascade. */
    void draw_shadow_geometry(const std::size_t cascade) {
        const std::span<const std::size_t> visible_slots = shadow_visible_slots_[cascade];
        const VkDeviceSize offset = 0;
        if (!uses_chunk_remapping() && visible_slots.size() == draw_ranges_.size()) {
            vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
            vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(command_buffer_, index_count_, 1, 0, 0, 0);
            if (shadow_index_count_ > 0U) {
                vkCmdBindIndexBuffer(
                    command_buffer_, shadow_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(command_buffer_, shadow_index_count_, 1, 0, 0, 0);
            }
            return;
        }
        if (!has_chunk_overrides()) {
            vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
            const IndirectDrawSpan& opaque = shadow_opaque_indirect_[cascade];
            if (opaque.count > 0U) {
                vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexedIndirect(
                    command_buffer_,
                    indirect_buffer_,
                    opaque.offset,
                    opaque.count,
                    sizeof(VkDrawIndexedIndirectCommand));
            }
            const IndirectDrawSpan& cloud = shadow_cloud_indirect_[cascade];
            if (cloud.count > 0U) {
                vkCmdBindIndexBuffer(
                    command_buffer_, shadow_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexedIndirect(
                    command_buffer_,
                    indirect_buffer_,
                    cloud.offset,
                    cloud.count,
                    sizeof(VkDrawIndexedIndirectCommand));
            }
            return;
        }
        for (const std::size_t index : visible_slots) {
            if (chunk_overrides_[index].has_value()) {
                const ChunkMeshBinding& binding = *chunk_overrides_[index];
                const PackedMeshBufferSet& buffers = binding.owner->buffers;
                vkCmdBindVertexBuffers(
                    command_buffer_, 0, 1, &buffers.buffer, &buffers.vertex_offset);
                vkCmdBindIndexBuffer(
                    command_buffer_, buffers.buffer, buffers.index_offset, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(
                    command_buffer_, binding.range.index_count, 1, binding.range.first_index, 0, 0);
                if (binding.range.shadow_index_count > 0U) {
                    vkCmdBindIndexBuffer(
                        command_buffer_,
                        buffers.buffer,
                        buffers.shadow_index_offset,
                        VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(
                        command_buffer_,
                        binding.range.shadow_index_count,
                        1,
                        binding.range.first_shadow_index,
                        0,
                        0);
                }
            } else {
                const world::MeshDrawRange& range =
                    draw_ranges_[base_range_for_slot_[index]];
                vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
                vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(
                    command_buffer_, range.index_count, 1, range.first_index, 0, 0);
                if (range.shadow_index_count > 0U) {
                    vkCmdBindIndexBuffer(
                        command_buffer_, shadow_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(
                        command_buffer_,
                        range.shadow_index_count,
                        1,
                        range.first_shadow_index,
                        0,
                        0);
                }
            }
        }
    }

    /** Populates the scene depth target before any expensive fragment shading. */
    void record_depth_prepass() {
        VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth_attachment.imageView = depth_view_;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue.depthStencil = {1.0F, 0};

        const VkRect2D render_area{{0, 0}, swapchain_extent_};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = render_area;
        rendering.layerCount = 1;
        rendering.pDepthAttachment = &depth_attachment;
        vkCmdBeginRendering(command_buffer_, &rendering);
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_prepass_pipeline_);
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            world_pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
        vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchain_extent_.width);
        viewport.height = static_cast<float>(swapchain_extent_.height);
        viewport.minDepth = 0.0F;
        viewport.maxDepth = 1.0F;
        vkCmdSetViewport(command_buffer_, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer_, 0, 1, &render_area);
        draw_opaque_geometry(scene_visible_slots_);
        vkCmdEndRendering(command_buffer_);

        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.image = depth_image_;
        barrier.subresourceRange = depth_subresource_range(1);
        submit_image_barrier(barrier);
    }

    /** Selects the nearest double-sided cloud shell after opaque color is available. */
    void record_cloud_depth_prepass() {
        if (cloud_index_count_ == 0U && !has_chunk_overrides()) {
            return;
        }

        VkImageMemoryBarrier2 writable_depth{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        writable_depth.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        writable_depth.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        writable_depth.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        writable_depth.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        writable_depth.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        writable_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        writable_depth.image = depth_image_;
        writable_depth.subresourceRange = depth_subresource_range(1);
        submit_image_barrier(writable_depth);

        VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth_attachment.imageView = depth_view_;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        const VkRect2D render_area{{0, 0}, swapchain_extent_};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = render_area;
        rendering.layerCount = 1;
        rendering.pDepthAttachment = &depth_attachment;
        vkCmdBeginRendering(command_buffer_, &rendering);
        vkCmdBindPipeline(
            command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, cloud_depth_prepass_pipeline_);
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            world_pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
        vkCmdBindIndexBuffer(command_buffer_, cloud_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchain_extent_.width);
        viewport.height = static_cast<float>(swapchain_extent_.height);
        viewport.minDepth = 0.0F;
        viewport.maxDepth = 1.0F;
        vkCmdSetViewport(command_buffer_, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer_, 0, 1, &render_area);
        draw_transparent_geometry(true);
        vkCmdEndRendering(command_buffer_);
    }

    /** Transitions the complete shadow array to a depth attachment. */
    void transition_shadow_for_writes() {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = shadow_initialized_ ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                                   : VK_PIPELINE_STAGE_2_NONE;
        barrier.srcAccessMask = shadow_initialized_ ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = shadow_initialized_ ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
                                                : VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.image = shadow_image_;
        barrier.subresourceRange = depth_subresource_range(static_cast<std::uint32_t>(cascade_count));
        submit_image_barrier(barrier);
    }

    /** Emits one dynamic-rendering depth pass per cascade layer. */
    void record_shadow_passes() {
        const VkDeviceSize offset = 0;
        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            shadow_pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
        vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        VkViewport viewport{};
        viewport.width = static_cast<float>(shadow_resolution);
        viewport.height = static_cast<float>(shadow_resolution);
        viewport.minDepth = 0.0F;
        viewport.maxDepth = 1.0F;
        const VkRect2D scissor{{0, 0}, {shadow_resolution, shadow_resolution}};
        vkCmdSetViewport(command_buffer_, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer_, 0, 1, &scissor);
        vkCmdSetDepthBias(command_buffer_, 1.25F, 0.0F, 1.75F);

        for (std::int32_t cascade = 0; cascade < static_cast<std::int32_t>(cascade_count); ++cascade) {
            if (!shadow_cascade_updates_[static_cast<std::size_t>(cascade)]) {
                continue;
            }
            VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            depth_attachment.imageView = shadow_layer_views_[static_cast<std::size_t>(cascade)];
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue.depthStencil = {1.0F, 0};

            VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
            rendering.renderArea = scissor;
            rendering.layerCount = 1;
            rendering.pDepthAttachment = &depth_attachment;
            vkCmdBeginRendering(command_buffer_, &rendering);
            vkCmdPushConstants(
                command_buffer_,
                shadow_pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(cascade),
                &cascade);
            draw_shadow_geometry(static_cast<std::size_t>(cascade));
            vkCmdEndRendering(command_buffer_);
        }
    }

    /** Makes completed shadow depth writes visible to the fragment shader. */
    void transition_shadow_for_reads() {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        barrier.image = shadow_image_;
        barrier.subresourceRange = depth_subresource_range(static_cast<std::uint32_t>(cascade_count));
        submit_image_barrier(barrier);
    }

    /** Prepares the HDR color target and shared scene depth image. */
    void transition_scene_attachments() {
        std::array<VkImageMemoryBarrier2, 2> barriers{};
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = scene_color_initialized_
                                       ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                       : VK_PIPELINE_STAGE_2_NONE;
        barriers[0].srcAccessMask = scene_color_initialized_
                                        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                        : VK_ACCESS_2_NONE;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].oldLayout = scene_color_initialized_
                                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                    : VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].image = scene_color_image_;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.layerCount = 1;

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask = depth_initialized_ ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                                      : VK_PIPELINE_STAGE_2_NONE;
        barriers[1].srcAccessMask = depth_initialized_
                                        ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                        : VK_ACCESS_2_NONE;
        barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].oldLayout = depth_initialized_ ? VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL
                                                   : VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barriers[1].image = depth_image_;
        barriers[1].subresourceRange = depth_subresource_range(1);

        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
        dependency.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(command_buffer_, &dependency);
    }

    /** Renders the lit voxel mesh into the HDR scene target. */
    void record_world_pass() {
        VkRenderingAttachmentInfo color_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color_attachment.imageView = scene_color_view_;
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.clearValue.color = {
            {current_sky_color_.r, current_sky_color_.g, current_sky_color_.b, 1.0F}};

        VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth_attachment.imageView = depth_view_;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        const VkRect2D render_area{{0, 0}, swapchain_extent_};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = render_area;
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color_attachment;
        rendering.pDepthAttachment = &depth_attachment;
        vkCmdBeginRendering(command_buffer_, &rendering);

        vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, sky_pipeline_);
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            world_pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchain_extent_.width);
        viewport.height = static_cast<float>(swapchain_extent_.height);
        viewport.minDepth = 0.0F;
        viewport.maxDepth = 1.0F;
        vkCmdSetViewport(command_buffer_, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer_, 0, 1, &render_area);
        vkCmdDraw(command_buffer_, 3, 1, 0, 0);

        if (!current_underwater_) {
            vkCmdBindPipeline(
                command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, celestial_pipeline_);
            const CelestialBasis sun_basis = celestial_basis(current_sun_direction_);
            const CelestialPushConstants sun{
                .direction_and_half_size = glm::vec4(current_sun_direction_, 6.8F),
                .color = {1.0F, 0.91F, 0.60F, 1.0F},
                .world_right = glm::vec4(sun_basis.right, 0.0F),
                .world_up = glm::vec4(sun_basis.up, 0.0F),
            };
            vkCmdPushConstants(
                command_buffer_,
                world_pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(sun),
                &sun);
            vkCmdDraw(command_buffer_, 6, 1, 0, 0);
            const CelestialBasis moon_basis = celestial_basis(-current_sun_direction_);
            const CelestialPushConstants moon{
                .direction_and_half_size = glm::vec4(-current_sun_direction_, 5.6F),
                .color = {0.78F, 0.84F, 0.94F, 1.0F},
                .world_right = glm::vec4(moon_basis.right, 0.0F),
                .world_up = glm::vec4(moon_basis.up, 0.0F),
            };
            vkCmdPushConstants(
                command_buffer_,
                world_pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(moon),
                &moon);
            vkCmdDraw(command_buffer_, 6, 1, 0, 0);
        }

        vkCmdBindPipeline(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            wireframe_enabled_ ? wireframe_pipeline_ : world_pipeline_);
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            world_pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
        vkCmdBindIndexBuffer(command_buffer_, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdSetViewport(command_buffer_, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer_, 0, 1, &render_area);
        draw_opaque_geometry(scene_visible_slots_);
        if (wireframe_enabled_) {
            draw_transparent_geometry(true);
            draw_transparent_geometry(false);
        }
        vkCmdEndRendering(command_buffer_);
    }

    /** Blends the selected cloud shell and water over completed opaque color. */
    void record_transparent_pass() {
        std::array<VkImageMemoryBarrier2, 2> barriers{};
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].image = scene_color_image_;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.layerCount = 1;

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barriers[1].image = depth_image_;
        barriers[1].subresourceRange = depth_subresource_range(1);

        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
        dependency.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(command_buffer_, &dependency);

        VkRenderingAttachmentInfo color_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color_attachment.imageView = scene_color_view_;
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo depth_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depth_attachment.imageView = depth_view_;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        const VkRect2D render_area{{0, 0}, swapchain_extent_};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = render_area;
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color_attachment;
        rendering.pDepthAttachment = &depth_attachment;
        vkCmdBeginRendering(command_buffer_, &rendering);
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            world_pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffer_, 0, 1, &vertex_buffer_, &offset);
        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchain_extent_.width);
        viewport.height = static_cast<float>(swapchain_extent_.height);
        viewport.minDepth = 0.0F;
        viewport.maxDepth = 1.0F;
        vkCmdSetViewport(command_buffer_, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer_, 0, 1, &render_area);

        if (cloud_index_count_ > 0U || has_chunk_overrides()) {
            vkCmdBindPipeline(
                command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, cloud_pipeline_);
            draw_transparent_geometry(true);
        }
        if (transparent_index_count_ > 0U || has_chunk_overrides()) {
            vkCmdBindPipeline(
                command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, transparent_pipeline_);
            draw_transparent_geometry(false);
        }
        vkCmdEndRendering(command_buffer_);
    }

    /** Makes completed HDR color and depth available to full-screen effects. */
    void transition_scene_for_postprocess() {
        std::array<VkImageMemoryBarrier2, 2> barriers{};
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barriers[0].srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].image = scene_color_image_;
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.layerCount = 1;

        barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        barriers[1].srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barriers[1].oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
        barriers[1].image = depth_image_;
        barriers[1].subresourceRange = depth_subresource_range(1);

        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size());
        dependency.pImageMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(command_buffer_, &dependency);
    }

    /** Prepares the acquired swapchain image as the post-processing output target. */
    void transition_swapchain_for_postprocess(const std::uint32_t image_index) {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = swapchain_image_initialized_[image_index]
                                   ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                                   : VK_PIPELINE_STAGE_2_NONE;
        barrier.srcAccessMask = VK_ACCESS_2_NONE;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = swapchain_image_initialized_[image_index]
                                ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                : VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.image = swapchain_images_[image_index];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        submit_image_barrier(barrier);
    }

    /** Applies SSAO, bloom, aerial lighting, god rays, and water reflections. */
    void record_postprocess_pass(const std::uint32_t image_index) {
        VkRenderingAttachmentInfo color_attachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color_attachment.imageView = swapchain_views_[image_index];
        color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        const VkRect2D render_area{{0, 0}, swapchain_extent_};
        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = render_area;
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &color_attachment;
        vkCmdBeginRendering(command_buffer_, &rendering);
        vkCmdBindPipeline(
            command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS, postprocess_pipeline_);
        vkCmdBindDescriptorSets(
            command_buffer_,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            world_pipeline_layout_,
            0,
            1,
            &descriptor_set_,
            0,
            nullptr);
        VkViewport viewport{};
        viewport.width = static_cast<float>(swapchain_extent_.width);
        viewport.height = static_cast<float>(swapchain_extent_.height);
        viewport.minDepth = 0.0F;
        viewport.maxDepth = 1.0F;
        vkCmdSetViewport(command_buffer_, 0, 1, &viewport);
        vkCmdSetScissor(command_buffer_, 0, 1, &render_area);
        vkCmdDraw(command_buffer_, 3, 1, 0, 0);
        vkCmdEndRendering(command_buffer_);
    }

    /** Transitions the completed color target for queue presentation. */
    void transition_color_for_present(const std::uint32_t image_index) {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrier.dstAccessMask = VK_ACCESS_2_NONE;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.image = swapchain_images_[image_index];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        submit_image_barrier(barrier);
    }

    /** Submits one synchronization2 image barrier to the active command buffer. */
    void submit_image_barrier(const VkImageMemoryBarrier2& barrier) {
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(command_buffer_, &dependency);
    }

    /** Returns a depth-only subresource range with the requested layer count. */
    [[nodiscard]] static VkImageSubresourceRange depth_subresource_range(
        const std::uint32_t layers) noexcept {
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        range.levelCount = 1;
        range.layerCount = layers;
        return range;
    }

    /** Creates a buffer and binds memory with the requested properties. */
    void create_buffer(
        const VkDeviceSize size,
        const VkBufferUsageFlags usage,
        const VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& memory) const {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        require_success(vkCreateBuffer(device_, &info, nullptr, &buffer), "Creating a buffer");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, properties);
        try {
            require_success(vkAllocateMemory(device_, &allocation, nullptr, &memory), "Allocating buffer memory");
            require_success(vkBindBufferMemory(device_, buffer, memory, 0), "Binding buffer memory");
        } catch (...) {
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }
            vkDestroyBuffer(device_, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            throw;
        }
    }

    /** Uploads tightly packed RGBA mip levels into a shader image. */
    void copy_buffer_to_image(
        const VkBuffer source,
        const VkImage destination,
        const std::span<const AtlasMipLevel> levels) {
        const VkCommandBuffer command = begin_one_time_commands();

        VkImageMemoryBarrier2 to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        to_transfer.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        to_transfer.srcAccessMask = VK_ACCESS_2_NONE;
        to_transfer.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_transfer.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer.image = destination;
        to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_transfer.subresourceRange.levelCount = static_cast<std::uint32_t>(levels.size());
        to_transfer.subresourceRange.layerCount = 1;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &to_transfer;
        vkCmdPipelineBarrier2(command, &dependency);

        std::vector<VkBufferImageCopy> copies;
        copies.reserve(levels.size());
        for (std::size_t mip = 0U; mip < levels.size(); ++mip) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = static_cast<VkDeviceSize>(levels[mip].byte_offset);
            copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copy.imageSubresource.mipLevel = static_cast<std::uint32_t>(mip);
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent = {levels[mip].width, levels[mip].height, 1};
            copies.push_back(copy);
        }
        vkCmdCopyBufferToImage(
            command,
            source,
            destination,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(copies.size()),
            copies.data());

        VkImageMemoryBarrier2 to_shader{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        to_shader.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        to_shader.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_shader.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        to_shader.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader.image = destination;
        to_shader.subresourceRange = to_transfer.subresourceRange;
        dependency.pImageMemoryBarriers = &to_shader;
        vkCmdPipelineBarrier2(command, &dependency);
        end_one_time_commands(command);
    }

    /** Creates a device-local image with the requested mip count. */
    void create_image(
        const std::uint32_t width,
        const std::uint32_t height,
        const std::uint32_t layers,
        const VkFormat format,
        const VkImageUsageFlags usage,
        VkImage& image,
        VkDeviceMemory& memory,
        const std::uint32_t mip_levels = 1U) const {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {width, height, 1};
        info.mipLevels = mip_levels;
        info.arrayLayers = layers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        require_success(vkCreateImage(device_, &info, nullptr, &image), "Creating an image");

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = find_memory_type(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        try {
            require_success(vkAllocateMemory(device_, &allocation, nullptr, &memory), "Allocating image memory");
            require_success(vkBindImageMemory(device_, image, memory, 0), "Binding image memory");
        } catch (...) {
            if (memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, memory, nullptr);
                memory = VK_NULL_HANDLE;
            }
            vkDestroyImage(device_, image, nullptr);
            image = VK_NULL_HANDLE;
            throw;
        }
    }

    /** Creates a color, depth, or layered image view. */
    [[nodiscard]] VkImageView create_image_view(
        const VkImage image,
        const VkFormat format,
        const VkImageAspectFlags aspect,
        const VkImageViewType type,
        const std::uint32_t base_layer,
        const std::uint32_t layer_count,
        const std::uint32_t mip_levels = 1U) const {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image;
        info.viewType = type;
        info.format = format;
        info.subresourceRange.aspectMask = aspect;
        info.subresourceRange.levelCount = mip_levels;
        info.subresourceRange.baseArrayLayer = base_layer;
        info.subresourceRange.layerCount = layer_count;
        VkImageView view = VK_NULL_HANDLE;
        require_success(vkCreateImageView(device_, &info, nullptr, &view), "Creating an image view");
        return view;
    }

    /** Selects the first widely supported depth attachment format. */
    [[nodiscard]] VkFormat find_depth_format(const bool sampled) const {
        constexpr std::array candidates{
            VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
        for (const VkFormat format : candidates) {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(physical_device_, format, &properties);
            VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
            if (sampled) {
                required |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            }
            if ((properties.optimalTilingFeatures & required) == required) {
                return format;
            }
        }
        throw std::runtime_error("The GPU has no supported depth attachment format");
    }

    /** Finds memory compatible with both a resource and requested property flags. */
    [[nodiscard]] std::uint32_t find_memory_type(
        const std::uint32_t type_filter,
        const VkMemoryPropertyFlags properties) const {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
        for (std::uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
            const bool allowed = (type_filter & (1U << index)) != 0U;
            const bool matches =
                (memory_properties.memoryTypes[index].propertyFlags & properties) == properties;
            if (allowed && matches) {
                return index;
            }
        }
        throw std::runtime_error("No compatible Vulkan memory type was found");
    }

    /** Begins a temporary primary command buffer for initialization transfers. */
    [[nodiscard]] VkCommandBuffer begin_one_time_commands() const {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        require_success(vkAllocateCommandBuffers(device_, &allocation, &command),
                        "Allocating a transfer command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        require_success(vkBeginCommandBuffer(command, &begin), "Beginning a transfer command buffer");
        return command;
    }

    /** Submits, waits for, and frees a temporary initialization command buffer. */
    void end_one_time_commands(const VkCommandBuffer command) const {
        require_success(vkEndCommandBuffer(command), "Ending a transfer command buffer");
        VkCommandBufferSubmitInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        command_info.commandBuffer = command;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &command_info;
        require_success(vkQueueSubmit2(graphics_queue_, 1, &submit, VK_NULL_HANDLE),
                        "Submitting a transfer command buffer");
        require_success(vkQueueWaitIdle(graphics_queue_), "Waiting for a transfer command buffer");
        vkFreeCommandBuffers(device_, command_pool_, 1, &command);
    }

    /** Rebuilds all resources whose size or format follows the window surface. */
    void recreate_swapchain() {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0) {
            glfwWaitEvents();
            glfwGetFramebufferSize(window_, &width, &height);
        }
        require_success(vkDeviceWaitIdle(device_), "Waiting before swapchain recreation");
        destroy_swapchain_resources();
        create_swapchain();
        create_depth_resources();
        create_scene_color_resources();
        update_descriptor_set();
        create_pipelines();
        create_render_finished_semaphores();
        framebuffer_resized_ = false;
    }

    /** Destroys pipelines, depth, views, and swapchain tied to surface extent. */
    void destroy_swapchain_resources() noexcept {
        if (device_ == VK_NULL_HANDLE) {
            return;
        }
        for (const VkSemaphore semaphore : render_finished_) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
        render_finished_.clear();
        vkDestroyPipeline(device_, postprocess_pipeline_, nullptr);
        postprocess_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, shadow_pipeline_, nullptr);
        shadow_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, sky_pipeline_, nullptr);
        sky_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, celestial_pipeline_, nullptr);
        celestial_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, depth_prepass_pipeline_, nullptr);
        depth_prepass_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, cloud_depth_prepass_pipeline_, nullptr);
        cloud_depth_prepass_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, cloud_pipeline_, nullptr);
        cloud_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, transparent_pipeline_, nullptr);
        transparent_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, wireframe_pipeline_, nullptr);
        wireframe_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipeline(device_, world_pipeline_, nullptr);
        world_pipeline_ = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(device_, shadow_pipeline_layout_, nullptr);
        shadow_pipeline_layout_ = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(device_, world_pipeline_layout_, nullptr);
        world_pipeline_layout_ = VK_NULL_HANDLE;
        vkDestroyImageView(device_, depth_view_, nullptr);
        depth_view_ = VK_NULL_HANDLE;
        vkDestroyImage(device_, depth_image_, nullptr);
        depth_image_ = VK_NULL_HANDLE;
        vkFreeMemory(device_, depth_memory_, nullptr);
        depth_memory_ = VK_NULL_HANDLE;
        vkDestroyImageView(device_, scene_color_view_, nullptr);
        scene_color_view_ = VK_NULL_HANDLE;
        vkDestroyImage(device_, scene_color_image_, nullptr);
        scene_color_image_ = VK_NULL_HANDLE;
        vkFreeMemory(device_, scene_color_memory_, nullptr);
        scene_color_memory_ = VK_NULL_HANDLE;
        for (const VkImageView view : swapchain_views_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        swapchain_views_.clear();
        swapchain_images_.clear();
        swapchain_image_initialized_.clear();
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }

    /** Releases all owned Vulkan handles and tolerates partial initialization. */
    void cleanup() noexcept {
        wait_idle();
        destroy_swapchain_resources();
        if (device_ != VK_NULL_HANDLE) {
            release_completed_packed_uploads(true);
            vkDestroyFence(device_, frame_fence_, nullptr);
            vkDestroySemaphore(device_, image_available_, nullptr);

            for (const VkImageView view : shadow_layer_views_) {
                vkDestroyImageView(device_, view, nullptr);
            }
            vkDestroyImageView(device_, shadow_array_view_, nullptr);
            vkDestroyImage(device_, shadow_image_, nullptr);
            vkFreeMemory(device_, shadow_memory_, nullptr);

            destroy_chunk_overrides();
            if (indirect_mapped_ != nullptr) {
                vkUnmapMemory(device_, indirect_memory_);
                indirect_mapped_ = nullptr;
            }
            vkDestroyBuffer(device_, indirect_buffer_, nullptr);
            vkFreeMemory(device_, indirect_memory_, nullptr);
            vkDestroyBuffer(device_, shadow_index_buffer_, nullptr);
            vkFreeMemory(device_, shadow_index_memory_, nullptr);
            vkDestroyBuffer(device_, index_buffer_, nullptr);
            vkFreeMemory(device_, index_memory_, nullptr);
            vkDestroyBuffer(device_, transparent_index_buffer_, nullptr);
            vkFreeMemory(device_, transparent_index_memory_, nullptr);
            vkDestroyBuffer(device_, cloud_index_buffer_, nullptr);
            vkFreeMemory(device_, cloud_index_memory_, nullptr);
            vkDestroyBuffer(device_, vertex_buffer_, nullptr);
            vkFreeMemory(device_, vertex_memory_, nullptr);
            if (uniform_mapped_ != nullptr) {
                vkUnmapMemory(device_, uniform_memory_);
            }
            vkDestroyBuffer(device_, uniform_buffer_, nullptr);
            vkFreeMemory(device_, uniform_memory_, nullptr);

            vkDestroyImageView(device_, atlas_view_, nullptr);
            vkDestroyImage(device_, atlas_image_, nullptr);
            vkFreeMemory(device_, atlas_memory_, nullptr);
            vkDestroySampler(device_, atlas_sampler_, nullptr);
            vkDestroySampler(device_, shadow_sampler_, nullptr);
            vkDestroySampler(device_, scene_sampler_, nullptr);
            vkDestroySampler(device_, depth_sampler_, nullptr);
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
            vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
            vkDestroyCommandPool(device_, command_pool_, nullptr);
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (surface_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
            surface_ = VK_NULL_HANDLE;
        }
        if (debug_messenger_ != VK_NULL_HANDLE) {
            const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy != nullptr) {
                destroy(instance_, debug_messenger_, nullptr);
            }
            debug_messenger_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
    }

    GLFWwindow* window_{nullptr};
    bool validation_enabled_{false};
    std::atomic_bool validation_error_seen_{false};
    bool framebuffer_resized_{false};
    bool shadow_initialized_{false};
    bool depth_initialized_{false};
    bool scene_color_initialized_{false};
    bool wireframe_enabled_{false};
    std::uint64_t frame_sequence_{};
    std::array<bool, cascade_count> shadow_cascade_updates_{};
    std::array<glm::mat4, cascade_count> cached_shadow_matrices_{};

    VkInstance instance_{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
    VkSurfaceKHR surface_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    QueueFamilies queue_families_;
    VkQueue graphics_queue_{VK_NULL_HANDLE};
    VkQueue present_queue_{VK_NULL_HANDLE};

    VkSwapchainKHR swapchain_{VK_NULL_HANDLE};
    VkFormat swapchain_format_{VK_FORMAT_UNDEFINED};
    VkExtent2D swapchain_extent_{};
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> swapchain_views_;
    std::vector<bool> swapchain_image_initialized_;

    VkCommandPool command_pool_{VK_NULL_HANDLE};
    VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
    VkSemaphore image_available_{VK_NULL_HANDLE};
    std::vector<VkSemaphore> render_finished_;
    VkFence frame_fence_{VK_NULL_HANDLE};

    VkBuffer vertex_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory vertex_memory_{VK_NULL_HANDLE};
    VkBuffer index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory index_memory_{VK_NULL_HANDLE};
    std::uint32_t index_count_{};
    VkBuffer transparent_index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory transparent_index_memory_{VK_NULL_HANDLE};
    std::uint32_t transparent_index_count_{};
    VkBuffer cloud_index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory cloud_index_memory_{VK_NULL_HANDLE};
    std::uint32_t cloud_index_count_{};
    VkBuffer shadow_index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory shadow_index_memory_{VK_NULL_HANDLE};
    std::uint32_t shadow_index_count_{};
    std::vector<world::MeshDrawRange> draw_ranges_;
    std::vector<std::optional<ChunkMeshBinding>> chunk_overrides_;
    std::vector<std::size_t> base_range_for_slot_;
    std::vector<PendingPackedUpload> pending_packed_uploads_;
    std::vector<world::ChunkCoordinate> active_chunk_coordinates_;
    std::vector<std::size_t> scene_visible_slots_;
    std::array<std::vector<std::size_t>, cascade_count> shadow_visible_slots_;
    VkBuffer indirect_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory indirect_memory_{VK_NULL_HANDLE};
    void* indirect_mapped_{nullptr};
    std::size_t indirect_command_capacity_{};
    std::vector<VkDrawIndexedIndirectCommand> indirect_commands_;
    IndirectDrawSpan opaque_indirect_;
    IndirectDrawSpan transparent_indirect_;
    IndirectDrawSpan cloud_indirect_;
    std::array<IndirectDrawSpan, cascade_count> shadow_opaque_indirect_;
    std::array<IndirectDrawSpan, cascade_count> shadow_cloud_indirect_;
    VkBuffer uniform_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory uniform_memory_{VK_NULL_HANDLE};
    void* uniform_mapped_{nullptr};

    VkDescriptorSetLayout descriptor_set_layout_{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    VkDescriptorSet descriptor_set_{VK_NULL_HANDLE};
    VkSampler shadow_sampler_{VK_NULL_HANDLE};
    VkSampler atlas_sampler_{VK_NULL_HANDLE};
    VkSampler scene_sampler_{VK_NULL_HANDLE};
    VkSampler depth_sampler_{VK_NULL_HANDLE};

    VkImage atlas_image_{VK_NULL_HANDLE};
    VkDeviceMemory atlas_memory_{VK_NULL_HANDLE};
    VkImageView atlas_view_{VK_NULL_HANDLE};

    VkFormat shadow_format_{VK_FORMAT_UNDEFINED};
    VkImage shadow_image_{VK_NULL_HANDLE};
    VkDeviceMemory shadow_memory_{VK_NULL_HANDLE};
    VkImageView shadow_array_view_{VK_NULL_HANDLE};
    std::array<VkImageView, cascade_count> shadow_layer_views_{};

    VkFormat depth_format_{VK_FORMAT_UNDEFINED};
    VkImage depth_image_{VK_NULL_HANDLE};
    VkDeviceMemory depth_memory_{VK_NULL_HANDLE};
    VkImageView depth_view_{VK_NULL_HANDLE};

    VkImage scene_color_image_{VK_NULL_HANDLE};
    VkDeviceMemory scene_color_memory_{VK_NULL_HANDLE};
    VkImageView scene_color_view_{VK_NULL_HANDLE};

    VkPipelineLayout world_pipeline_layout_{VK_NULL_HANDLE};
    VkPipelineLayout shadow_pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline world_pipeline_{VK_NULL_HANDLE};
    VkPipeline wireframe_pipeline_{VK_NULL_HANDLE};
    VkPipeline transparent_pipeline_{VK_NULL_HANDLE};
    VkPipeline cloud_pipeline_{VK_NULL_HANDLE};
    VkPipeline depth_prepass_pipeline_{VK_NULL_HANDLE};
    VkPipeline cloud_depth_prepass_pipeline_{VK_NULL_HANDLE};
    VkPipeline sky_pipeline_{VK_NULL_HANDLE};
    VkPipeline celestial_pipeline_{VK_NULL_HANDLE};
    VkPipeline postprocess_pipeline_{VK_NULL_HANDLE};
    VkPipeline shadow_pipeline_{VK_NULL_HANDLE};
    glm::vec3 current_sky_color_{0.20F, 0.46F, 0.78F};
    glm::vec3 current_sun_direction_{0.0F, 1.0F, 0.0F};
    bool current_underwater_{false};
    bool console_visible_{false};
    std::string console_text_;
    CascadeShadowCalculator cascade_calculator_;
};

VulkanRenderer::VulkanRenderer(
    GLFWwindow* window,
    const world::WorldMesh& mesh,
    const std::span<const world::ChunkCoordinate> chunk_coordinates)
    : impl_(std::make_unique<Impl>(window, mesh, chunk_coordinates)) {}

VulkanRenderer::~VulkanRenderer() = default;

void VulkanRenderer::draw_frame(
    const core::Camera& camera,
    const float world_time_seconds,
    const float animation_seconds,
    const bool underwater) {
    impl_->draw_frame(camera, world_time_seconds, animation_seconds, underwater);
}

void VulkanRenderer::update_world_mesh(const world::WorldMesh& mesh) {
    impl_->update_world_mesh(mesh);
}

void VulkanRenderer::update_world_chunk(
    const std::size_t chunk_index,
    const world::WorldMesh& mesh) {
    impl_->update_world_chunk(chunk_index, mesh);
}

void VulkanRenderer::update_streamed_chunks(
    const std::span<const world::ChunkCoordinate> chunk_coordinates,
    const std::span<const std::size_t> replacement_indices,
    const world::WorldMesh& replacement_mesh) {
    impl_->update_streamed_chunks(
        chunk_coordinates, replacement_indices, replacement_mesh);
}

void VulkanRenderer::set_wireframe_enabled(const bool enabled) noexcept {
    impl_->set_wireframe_enabled(enabled);
}

void VulkanRenderer::set_console_overlay(
    const bool visible,
    const std::string_view text) noexcept {
    impl_->set_console_overlay(visible, text);
}

void VulkanRenderer::notify_framebuffer_resized() noexcept {
    impl_->notify_framebuffer_resized();
}

void VulkanRenderer::wait_idle() const noexcept {
    impl_->wait_idle();
}

bool VulkanRenderer::has_validation_errors() const noexcept {
    return impl_->has_validation_errors();
}

}  // namespace vulkancraft::rendering
