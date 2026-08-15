#include "vulkancraft/core/Application.hpp"

#define NOMINMAX
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <Windows.h>

#include <future>
#include <iostream>
#include <thread>

#include "vulkancraft/rendering/CelestialCycle.hpp"
#include "vulkancraft/rendering/VulkanRenderer.hpp"

namespace vulkancraft::core {
namespace {

constexpr int window_width = 1280;
constexpr int window_height = 720;

/** Paints a responsive native loading screen before Vulkan is initialized. */
void paint_loading_screen(GLFWwindow* window) noexcept {
    const HWND native_window = glfwGetWin32Window(window);
    if (native_window == nullptr) {
        return;
    }
    RECT client{};
    if (GetClientRect(native_window, &client) == FALSE) {
        return;
    }
    const HDC device_context = GetDC(native_window);
    if (device_context == nullptr) {
        return;
    }
    const HBRUSH background = CreateSolidBrush(RGB(18, 27, 42));
    FillRect(device_context, &client, background);
    DeleteObject(background);
    SetBkMode(device_context, TRANSPARENT);
    SetTextColor(device_context, RGB(235, 242, 255));
    HFONT font = CreateFontW(
        42,
        0,
        0,
        0,
        FW_SEMIBOLD,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        L"Segoe UI");
    const HGDIOBJ previous_font = SelectObject(device_context, font);
    DrawTextW(
        device_context,
        L"VulkanCraft\nLoading world...",
        -1,
        &client,
        DT_CENTER | DT_VCENTER | DT_WORDBREAK);
    SelectObject(device_context, previous_font);
    DeleteObject(font);
    ReleaseDC(native_window, device_context);
}

}  // namespace

Application::Application(
    const bool window_visible,
    const bool start_wireframe,
    const bool start_console,
    const bool start_effects_view) {
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    try {
        if (glfwVulkanSupported() != GLFW_TRUE) {
            throw std::runtime_error("GLFW could not find a Vulkan loader");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, window_visible ? GLFW_TRUE : GLFW_FALSE);
        glfwWindowHint(GLFW_MAXIMIZED, window_visible ? GLFW_TRUE : GLFW_FALSE);
        window_ = glfwCreateWindow(window_width, window_height, "VulkanCraft", nullptr, nullptr);
        if (window_ == nullptr) {
            throw std::runtime_error("Failed to create the VulkanCraft window");
        }

        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
        glfwSetCursorPosCallback(window_, cursor_position_callback);
        glfwSetCharCallback(window_, character_callback);
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if (glfwRawMouseMotionSupported() == GLFW_TRUE) {
            glfwSetInputMode(window_, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        }

        glfwSetWindowTitle(window_, "VulkanCraft - Loading world...");
        auto initial_world = std::async(std::launch::async, [this] {
            return streaming_world_.initialize(camera_.position());
        });
        while (initial_world.wait_for(std::chrono::milliseconds{0}) !=
               std::future_status::ready) {
            glfwPollEvents();
            if (glfwWindowShouldClose(window_) == GLFW_TRUE) {
                throw std::runtime_error("The loading window was closed");
            }
            if (window_visible) {
                paint_loading_screen(window_);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{16});
        }
        auto mesh = initial_world.get();
        const auto chunk_coordinates = streaming_world_.visible_chunk_coordinates();
        player_.spawn(streaming_world_, camera_.position().x, camera_.position().z);
        if (start_effects_view) {
            std::optional<glm::ivec3> water_surface;
            const int origin_x = static_cast<int>(std::floor(camera_.position().x));
            const int origin_z = static_cast<int>(std::floor(camera_.position().z));
            for (int radius = 16; radius <= 320 && !water_surface.has_value(); radius += 8) {
                for (int offset_z = -radius; offset_z <= radius && !water_surface.has_value();
                     offset_z += 4) {
                    for (int offset_x = -radius; offset_x <= radius; offset_x += 4) {
                        if (std::abs(offset_x) != radius && std::abs(offset_z) != radius) {
                            continue;
                        }
                        const int x = origin_x + offset_x;
                        const int z = origin_z + offset_z;
                        for (int y = 68; y >= 58; --y) {
                            if (streaming_world_.block_at(x, y, z) == world::BlockType::water &&
                                streaming_world_.block_at(x + 3, y, z) == world::BlockType::water &&
                                streaming_world_.block_at(x - 3, y, z) == world::BlockType::water &&
                                streaming_world_.block_at(x, y, z + 3) == world::BlockType::water &&
                                streaming_world_.block_at(x, y + 1, z) == world::BlockType::air) {
                                water_surface = glm::ivec3{x, y, z};
                                break;
                            }
                        }
                        if (water_surface.has_value()) {
                            break;
                        }
                    }
                }
            }
            if (water_surface.has_value()) {
                player_.set_position({
                    static_cast<float>(water_surface->x) + 0.5F,
                    static_cast<float>(water_surface->y) + 5.0F,
                    static_cast<float>(water_surface->z) + 20.5F});
                player_.update(
                    {.jump = true}, camera_.forward(), 0.01F, streaming_world_);
                player_.update({}, camera_.forward(), 0.05F, streaming_world_);
                player_.update(
                    {.jump = true}, camera_.forward(), 0.01F, streaming_world_);
                camera_.look(0.0F, -100.0F);
                world_time_seconds_ = 55.0F;
            }
        }
        camera_.set_position(player_.eye_position());
        renderer_ = std::make_unique<rendering::VulkanRenderer>(
            window_, mesh, chunk_coordinates);
        wireframe_enabled_ = start_wireframe;
        renderer_->set_wireframe_enabled(wireframe_enabled_);
        console_open_ = start_console;
        if (console_open_) {
            console_input_ = "/time set night";
            glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        refresh_window_title();
    } catch (...) {
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }
        glfwTerminate();
        throw;
    }
}

Application::~Application() {
    renderer_.reset();
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

void Application::run(
    const std::optional<std::uint32_t> frame_limit,
    const bool exercise_streaming,
    const bool exercise_editing,
    const float elapsed_time_offset) {
    using clock = std::chrono::steady_clock;
    const auto started_at = clock::now();
    auto previous_frame = started_at;
    std::uint32_t rendered_frames = 0;
    std::uint32_t completed_streams = 0;
    constexpr std::uint32_t streaming_smoke_steps = 12;
    std::optional<clock::time_point> edit_requested_at;
    std::optional<clock::time_point> stream_requested_at;

    while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
        glfwPollEvents();
        if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }

        const auto now = clock::now();
        const float delta_seconds = std::clamp(
            std::chrono::duration<float>(now - previous_frame).count(), 0.0F, 0.1F);
        const float elapsed_seconds = std::chrono::duration<float>(now - started_at).count();
        previous_frame = now;

        update_debug_controls();
        update_window_controls();
        world_time_seconds_ += delta_seconds;
        if (!console_open_) {
            player_.update(read_player_input(), camera_.forward(), delta_seconds, streaming_world_);
        }
        camera_.set_position(player_.eye_position());
        if (!console_open_) {
            update_block_interactions();
        }
        streaming_world_.advance_water(delta_seconds);
        const auto dirty_chunks = streaming_world_.consume_dirty_chunk_indices();
        for (const std::size_t chunk_index : dirty_chunks) {
            renderer_->update_world_chunk(chunk_index, streaming_world_.chunk_mesh(chunk_index));
        }
        if (!dirty_chunks.empty() && edit_requested_at.has_value()) {
            const auto edit_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                clock::now() - *edit_requested_at);
            std::cout << "Block edit visible after " << edit_latency.count() << " ms.\n";
            edit_requested_at.reset();
        }
        if (auto streamed_update = streaming_world_.update(player_.position());
            streamed_update.has_value()) {
            const auto install_started_at = clock::now();
            const auto streamed_coordinates = streaming_world_.visible_chunk_coordinates();
            renderer_->update_streamed_chunks(
                streamed_coordinates,
                streamed_update->replacement_indices,
                streamed_update->mesh);
            streaming_world_.retire_update(std::move(*streamed_update));
            if (stream_requested_at.has_value()) {
                const auto install_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                    clock::now() - install_started_at);
                const auto stream_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
                    clock::now() - *stream_requested_at);
                std::cout << "GPU chunk-window install completed in "
                          << install_latency.count() << " ms.\n";
                std::cout << "Streamed chunk window visible after " << stream_latency.count()
                          << " ms.\n";
                stream_requested_at.reset();
                ++completed_streams;
                if (exercise_streaming && completed_streams < streaming_smoke_steps) {
                    stream_requested_at = clock::now();
                    player_.set_position(
                        player_.position() + glm::vec3{33.0F, 0.0F, 0.0F});
                    camera_.set_position(player_.eye_position());
                }
            }
        }
        const glm::vec3 eye_position = player_.eye_position();
        const bool camera_submerged = streaming_world_.block_at(
            static_cast<int>(std::floor(eye_position.x)),
            static_cast<int>(std::floor(eye_position.y)),
            static_cast<int>(std::floor(eye_position.z))) == world::BlockType::water;
        renderer_->draw_frame(
            camera_, world_time_seconds_ + elapsed_time_offset, elapsed_seconds, camera_submerged);
        ++rendered_frames;
        if (exercise_streaming && rendered_frames == 1U) {
            stream_requested_at = clock::now();
            player_.set_position(player_.position() + glm::vec3{33.0F, 0.0F, 0.0F});
            camera_.set_position(player_.eye_position());
        }
        if (exercise_editing && rendered_frames == 1U) {
            const glm::ivec3 floor_block{
                static_cast<int>(std::floor(player_.position().x)),
                static_cast<int>(std::floor(player_.position().y)) - 1,
                static_cast<int>(std::floor(player_.position().z))};
            edit_requested_at = clock::now();
            if (!streaming_world_.set_block(
                    floor_block.x, floor_block.y, floor_block.z, world::BlockType::air)) {
                edit_requested_at.reset();
            }
        }
        if (exercise_streaming && completed_streams >= streaming_smoke_steps) {
            break;
        }
        if (frame_limit.has_value() && rendered_frames >= *frame_limit) {
            break;
        }
    }
    renderer_->wait_idle();
    if (exercise_streaming && stream_requested_at.has_value()) {
        throw std::runtime_error("Streaming smoke test timed out before installing a chunk window");
    }
    if (renderer_->has_validation_errors()) {
        throw std::runtime_error("Vulkan validation reported one or more errors");
    }
}

void Application::framebuffer_size_callback(
    GLFWwindow* window,
    const int width,
    const int height) noexcept {
    static_cast<void>(width);
    static_cast<void>(height);
    auto* application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (application != nullptr && application->renderer_ != nullptr) {
        application->renderer_->notify_framebuffer_resized();
    }
}

void Application::cursor_position_callback(
    GLFWwindow* window,
    const double x,
    const double y) noexcept {
    auto* application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (application == nullptr || application->console_open_) {
        return;
    }
    if (!application->received_first_cursor_event_) {
        application->previous_cursor_x_ = x;
        application->previous_cursor_y_ = y;
        application->received_first_cursor_event_ = true;
        return;
    }

    const auto delta_x = static_cast<float>(x - application->previous_cursor_x_);
    const auto delta_y = static_cast<float>(y - application->previous_cursor_y_);
    application->previous_cursor_x_ = x;
    application->previous_cursor_y_ = y;
    application->camera_.look(delta_x, delta_y);
}

void Application::character_callback(
    GLFWwindow* window,
    const unsigned int codepoint) noexcept {
    auto* application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (application == nullptr || !application->console_open_ || codepoint < 32U ||
        codepoint > 126U || codepoint == static_cast<unsigned int>('\'')) {
        return;
    }
    if (application->console_input_.size() < 96U) {
        application->console_input_.push_back(static_cast<char>(codepoint));
        application->console_message_.clear();
        application->refresh_window_title();
    }
}

PlayerInput Application::read_player_input() const noexcept {
    const auto axis = [this](const int positive, const int negative) {
        const float positive_value = glfwGetKey(window_, positive) == GLFW_PRESS ? 1.0F : 0.0F;
        const float negative_value = glfwGetKey(window_, negative) == GLFW_PRESS ? 1.0F : 0.0F;
        return positive_value - negative_value;
    };
    return {
        .forward = axis(GLFW_KEY_W, GLFW_KEY_S),
        .right = axis(GLFW_KEY_D, GLFW_KEY_A),
        .jump = glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS,
        .sprint = glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS,
        .crouch = glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS,
    };
}

void Application::update_block_interactions() noexcept {
    const bool primary_pressed = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool secondary_pressed = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    const bool break_requested = primary_pressed && !primary_mouse_was_pressed_;
    const bool place_requested = secondary_pressed && !secondary_mouse_was_pressed_;
    primary_mouse_was_pressed_ = primary_pressed;
    secondary_mouse_was_pressed_ = secondary_pressed;
    if (!break_requested && !place_requested) {
        return;
    }

    const glm::vec3 origin = camera_.position();
    const glm::vec3 direction = camera_.forward();
    glm::ivec3 previous_block = glm::ivec3(glm::floor(origin));
    for (float distance = 0.05F; distance <= 6.0F; distance += 0.05F) {
        const glm::ivec3 block = glm::ivec3(glm::floor(origin + direction * distance));
        if (block == previous_block) {
            continue;
        }
        const world::BlockType material = streaming_world_.block_at(block.x, block.y, block.z);
        if (world::is_renderable(material) && material != world::BlockType::cloud) {
            if (break_requested) {
                static_cast<void>(streaming_world_.set_block(
                    block.x, block.y, block.z, world::BlockType::air));
            } else if (!player_.occupies_block(previous_block.x, previous_block.y, previous_block.z)) {
                const world::BlockType placement_target = streaming_world_.block_at(
                    previous_block.x, previous_block.y, previous_block.z);
                if (!world::is_solid(placement_target)) {
                    static_cast<void>(streaming_world_.set_block(
                        previous_block.x,
                        previous_block.y,
                        previous_block.z,
                        world::BlockType::dirt));
                }
            }
            return;
        }
        previous_block = block;
    }
}

void Application::update_debug_controls() noexcept {
    const bool f3_pressed = glfwGetKey(window_, GLFW_KEY_F3) == GLFW_PRESS;
    if (f3_pressed && !f3_was_pressed_) {
        wireframe_enabled_ = !wireframe_enabled_;
        renderer_->set_wireframe_enabled(wireframe_enabled_);
    }
    f3_was_pressed_ = f3_pressed;
}

void Application::update_window_controls() noexcept {
    const bool f11_pressed = glfwGetKey(window_, GLFW_KEY_F11) == GLFW_PRESS;
    if (f11_pressed && !f11_was_pressed_) {
        set_borderless_fullscreen(!borderless_fullscreen_);
    }
    f11_was_pressed_ = f11_pressed;

    const bool toggle_pressed = glfwGetKey(window_, GLFW_KEY_APOSTROPHE) == GLFW_PRESS;
    if (toggle_pressed && !console_toggle_was_pressed_) {
        console_open_ = !console_open_;
        received_first_cursor_event_ = false;
        glfwSetInputMode(
            window_, GLFW_CURSOR, console_open_ ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        refresh_window_title();
    }
    console_toggle_was_pressed_ = toggle_pressed;
    if (!console_open_) {
        return;
    }

    const bool backspace_pressed = glfwGetKey(window_, GLFW_KEY_BACKSPACE) == GLFW_PRESS;
    if (backspace_pressed && !console_backspace_was_pressed_ && !console_input_.empty()) {
        console_input_.pop_back();
        console_message_.clear();
        refresh_window_title();
    }
    console_backspace_was_pressed_ = backspace_pressed;
    const bool enter_pressed = glfwGetKey(window_, GLFW_KEY_ENTER) == GLFW_PRESS ||
                               glfwGetKey(window_, GLFW_KEY_KP_ENTER) == GLFW_PRESS;
    if (enter_pressed && !console_enter_was_pressed_) {
        execute_console_command();
    }
    console_enter_was_pressed_ = enter_pressed;
}

void Application::execute_console_command() {
    if (console_input_ == "/time set day") {
        world_time_seconds_ = rendering::daylight_duration_seconds * 0.5F;
        console_message_ = "Time set to day";
    } else if (console_input_ == "/time set night") {
        world_time_seconds_ = rendering::daylight_duration_seconds * 1.5F;
        console_message_ = "Time set to night";
    } else if (!console_input_.empty()) {
        console_message_ = "Unknown command: " + console_input_;
    }
    console_input_.clear();
    refresh_window_title();
}

void Application::refresh_window_title() noexcept {
    if (!console_open_) {
        glfwSetWindowTitle(window_, "VulkanCraft");
        if (renderer_ != nullptr) {
            renderer_->set_console_overlay(false, {});
        }
        return;
    }
    const std::string prompt = console_input_.empty() && !console_message_.empty()
                                   ? "VulkanCraft Console - " + console_message_
                                   : "VulkanCraft Console > " + console_input_;
    glfwSetWindowTitle(window_, prompt.c_str());
    if (renderer_ != nullptr) {
        const std::string overlay_text = console_input_.empty() && !console_message_.empty()
                                             ? console_message_
                                             : "> " + console_input_;
        renderer_->set_console_overlay(true, overlay_text);
    }
}

void Application::set_borderless_fullscreen(const bool enabled) noexcept {
    if (enabled == borderless_fullscreen_) {
        return;
    }
    if (enabled) {
        window_was_maximized_ = glfwGetWindowAttrib(window_, GLFW_MAXIMIZED) == GLFW_TRUE;
        glfwGetWindowPos(window_, &windowed_x_, &windowed_y_);
        glfwGetWindowSize(window_, &windowed_width_, &windowed_height_);
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = monitor == nullptr ? nullptr : glfwGetVideoMode(monitor);
        if (monitor == nullptr || mode == nullptr) {
            return;
        }
        int monitor_x = 0;
        int monitor_y = 0;
        glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);
        glfwRestoreWindow(window_);
        glfwSetWindowAttrib(window_, GLFW_DECORATED, GLFW_FALSE);
        glfwSetWindowMonitor(
            window_, nullptr, monitor_x, monitor_y, mode->width, mode->height, GLFW_DONT_CARE);
    } else {
        glfwSetWindowAttrib(window_, GLFW_DECORATED, GLFW_TRUE);
        glfwSetWindowMonitor(
            window_,
            nullptr,
            windowed_x_,
            windowed_y_,
            windowed_width_,
            windowed_height_,
            GLFW_DONT_CARE);
        if (window_was_maximized_) {
            glfwMaximizeWindow(window_);
        }
    }
    borderless_fullscreen_ = enabled;
}

}  // namespace vulkancraft::core
