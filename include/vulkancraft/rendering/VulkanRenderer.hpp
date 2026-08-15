#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

struct GLFWwindow;

namespace vulkancraft::core {
class Camera;
}

namespace vulkancraft::world {
struct ChunkCoordinate;
struct WorldMesh;
}

namespace vulkancraft::rendering {

/** Owns Vulkan resources and renders the immutable world mesh. */
class VulkanRenderer final {
public:
    /**
     * Initializes Vulkan and uploads a world mesh.
     *
     * @param window Live GLFW window created without a client API.
     * @param mesh Non-empty voxel mesh to upload.
     * @param chunk_coordinates Global coordinate for every row-major mesh range.
     * @throws std::runtime_error If required Vulkan capabilities are unavailable.
     */
    VulkanRenderer(
        GLFWwindow* window,
        const world::WorldMesh& mesh,
        std::span<const world::ChunkCoordinate> chunk_coordinates);

    /** Releases all GPU resources after waiting for outstanding work. */
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;
    VulkanRenderer(VulkanRenderer&&) = delete;
    VulkanRenderer& operator=(VulkanRenderer&&) = delete;

    /**
     * Renders and presents one frame.
     *
     * @param camera Current player camera.
     * @param world_time_seconds Accumulated world-clock time used by the celestial cycle.
     * @param animation_seconds Monotonic application time used for cloud animation.
     * @param underwater Whether the camera is currently submerged.
     */
    void draw_frame(
        const core::Camera& camera,
        float world_time_seconds,
        float animation_seconds,
        bool underwater);

    /**
     * Replaces the GPU-resident world geometry after a streaming update.
     *
     * @param mesh Non-empty replacement mesh using global world coordinates.
     */
    void update_world_mesh(const world::WorldMesh& mesh);

    /**
     * Replaces one GPU-resident chunk batch after a local block edit.
     *
     * @param chunk_index Stable index inside the active streamed window.
     * @param mesh Replacement chunk mesh using global world coordinates.
     */
    void update_world_chunk(std::size_t chunk_index, const world::WorldMesh& mesh);

    /**
     * Reuses overlapping GPU chunks and installs one packed streaming replacement batch.
     *
     * @param chunk_coordinates New row-major global chunk coordinates.
     * @param replacement_indices Row-major slots replaced by the packed batch.
     * @param replacement_mesh Precombined entering and LOD-transition chunk ranges.
     */
    void update_streamed_chunks(
        std::span<const world::ChunkCoordinate> chunk_coordinates,
        std::span<const std::size_t> replacement_indices,
        const world::WorldMesh& replacement_mesh);

    /**
     * Selects the solid or wireframe world pipeline.
     *
     * @param enabled True to draw polygon edges only.
     */
    void set_wireframe_enabled(bool enabled) noexcept;

    /**
     * Updates the in-game cheat-console overlay.
     *
     * @param visible Whether the translucent console panel is visible.
     * @param text ASCII prompt or result text displayed by the panel.
     */
    void set_console_overlay(bool visible, std::string_view text) noexcept;

    /** Marks swapchain resources for recreation before the next usable frame. */
    void notify_framebuffer_resized() noexcept;

    /** Waits until the Vulkan device has completed all submitted work. */
    void wait_idle() const noexcept;

    /**
     * Reports whether validation emitted an error during this run.
     *
     * @return True when at least one validation error was observed.
     */
    [[nodiscard]] bool has_validation_errors() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vulkancraft::rendering
