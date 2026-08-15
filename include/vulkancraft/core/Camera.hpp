#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace vulkancraft::core {

/** Directional movement requested during one application frame. */
struct CameraMovement {
    float forward{};
    float right{};
    float upward{};
    bool sprint{};
};

/** Maintains a free-fly perspective camera. */
class Camera final {
public:
    /**
     * Creates a camera at the requested position.
     *
     * @param position Initial world-space position.
     */
    explicit Camera(glm::vec3 position = {48.0F, 88.0F, 70.0F}) noexcept;

    /**
     * Applies local movement for one frame.
     *
     * @param movement Normalized input intent.
     * @param delta_seconds Frame duration in seconds.
     */
    void move(const CameraMovement& movement, float delta_seconds) noexcept;

    /**
     * Rotates the camera from mouse motion.
     *
     * @param delta_x Horizontal cursor delta in pixels.
     * @param delta_y Vertical cursor delta in pixels.
     */
    void look(float delta_x, float delta_y) noexcept;

    /**
     * Repositions the camera without changing its orientation.
     *
     * @param position New world-space position.
     */
    void set_position(glm::vec3 position) noexcept;

    /**
     * Builds the right-handed view matrix.
     *
     * @return World-to-view transform.
     */
    [[nodiscard]] glm::mat4 view_matrix() const noexcept;

    /**
     * Builds a Vulkan-compatible perspective matrix.
     *
     * @param aspect_ratio Positive viewport width divided by height.
     * @return View-to-clip transform with zero-to-one depth.
     */
    [[nodiscard]] glm::mat4 projection_matrix(float aspect_ratio) const;

    /** Returns the current world-space position. */
    [[nodiscard]] const glm::vec3& position() const noexcept;

    /** Returns the normalized forward vector. */
    [[nodiscard]] glm::vec3 forward() const noexcept;

    /** Returns the vertical field of view in radians. */
    [[nodiscard]] float field_of_view() const noexcept;

    /** Returns the near clipping distance. */
    [[nodiscard]] float near_plane() const noexcept;

    /** Returns the far clipping distance. */
    [[nodiscard]] float far_plane() const noexcept;

private:
    glm::vec3 position_;
    float yaw_degrees_{-90.0F};
    float pitch_degrees_{-18.0F};
    float move_speed_{11.0F};
    float mouse_sensitivity_{0.10F};
    float field_of_view_radians_{1.0471975512F};
    float near_plane_{0.1F};
    float far_plane_{832.0F};
};

}  // namespace vulkancraft::core
