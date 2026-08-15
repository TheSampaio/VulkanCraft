#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "vulkancraft/core/Camera.hpp"
#include "vulkancraft/core/PlayerController.hpp"
#include "vulkancraft/world/StreamingWorld.hpp"

struct GLFWwindow;

namespace vulkancraft::rendering {
class VulkanRenderer;
}

namespace vulkancraft::core {

/** Coordinates the window, input, generated world, camera, and renderer. */
class Application final {
public:
    /**
     * Creates the game window, initial world, and Vulkan renderer.
     *
     * @param window_visible Whether the GLFW window should be shown.
     * @param start_wireframe Whether the first frame should use line polygons.
     * @param start_console Whether the visual console smoke test starts open.
     * @param start_effects_view Whether to stage the visual smoke test above nearby water.
     */
    explicit Application(
        bool window_visible = true,
        bool start_wireframe = false,
        bool start_console = false,
        bool start_effects_view = false);

    /** Releases the renderer before destroying GLFW resources. */
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    /**
     * Runs the event and rendering loop until the window closes.
     *
     * @param frame_limit Optional frame count used by automated smoke tests.
     * @param exercise_streaming Whether to cross a chunk boundary after frame one.
     * @param exercise_editing Whether to break the spawn-floor block after frame one.
     * @param elapsed_time_offset Seconds added to the day-night clock.
     */
    void run(
        std::optional<std::uint32_t> frame_limit = std::nullopt,
        bool exercise_streaming = false,
        bool exercise_editing = false,
        float elapsed_time_offset = 0.0F);

private:
    /** Receives framebuffer size notifications from GLFW. */
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) noexcept;

    /** Receives relative cursor motion from GLFW. */
    static void cursor_position_callback(GLFWwindow* window, double x, double y) noexcept;

    /** Receives printable Unicode input while the cheat console is open. */
    static void character_callback(GLFWwindow* window, unsigned int codepoint) noexcept;

    /** Translates the current keyboard state into player movement. */
    [[nodiscard]] PlayerInput read_player_input() const noexcept;

    /** Handles edge-triggered mouse actions for breaking and placing blocks. */
    void update_block_interactions() noexcept;

    /** Handles edge-triggered debug controls such as the wireframe toggle. */
    void update_debug_controls() noexcept;

    /** Handles borderless fullscreen and cheat-console controls. */
    void update_window_controls() noexcept;

    /** Executes one supported slash command from the cheat console. */
    void execute_console_command();

    /** Updates the native title with the active console prompt or game name. */
    void refresh_window_title() noexcept;

    /** Enters or leaves monitor-sized borderless window mode. */
    void set_borderless_fullscreen(bool enabled) noexcept;

    GLFWwindow* window_{nullptr};
    world::StreamingWorld streaming_world_;
    std::unique_ptr<rendering::VulkanRenderer> renderer_;
    Camera camera_;
    PlayerController player_;
    double previous_cursor_x_{};
    double previous_cursor_y_{};
    bool received_first_cursor_event_{false};
    bool f3_was_pressed_{false};
    bool f11_was_pressed_{false};
    bool console_toggle_was_pressed_{false};
    bool console_enter_was_pressed_{false};
    bool console_backspace_was_pressed_{false};
    bool wireframe_enabled_{false};
    bool borderless_fullscreen_{false};
    bool console_open_{false};
    bool window_was_maximized_{true};
    bool primary_mouse_was_pressed_{false};
    bool secondary_mouse_was_pressed_{false};
    int windowed_x_{};
    int windowed_y_{};
    int windowed_width_{1280};
    int windowed_height_{720};
    float world_time_seconds_{195.0F};
    std::string console_input_;
    std::string console_message_;
};

}  // namespace vulkancraft::core
