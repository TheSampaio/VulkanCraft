#include "vulkancraft/core/Camera.hpp"

#include <algorithm>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>

namespace vulkancraft::core {

Camera::Camera(const glm::vec3 position) noexcept : position_(position) {}

void Camera::move(const CameraMovement& movement, const float delta_seconds) noexcept {
    const glm::vec3 forward_flat = glm::normalize(glm::vec3{forward().x, 0.0F, forward().z});
    const glm::vec3 right = glm::normalize(glm::cross(forward_flat, glm::vec3{0.0F, 1.0F, 0.0F}));
    glm::vec3 direction = forward_flat * movement.forward + right * movement.right +
                          glm::vec3{0.0F, movement.upward, 0.0F};
    const float length = glm::length(direction);
    if (length > 1.0F) {
        direction /= length;
    }
    const float sprint_multiplier = movement.sprint ? 3.0F : 1.0F;
    position_ += direction * move_speed_ * sprint_multiplier * std::max(delta_seconds, 0.0F);
}

void Camera::look(const float delta_x, const float delta_y) noexcept {
    yaw_degrees_ += delta_x * mouse_sensitivity_;
    pitch_degrees_ = std::clamp(pitch_degrees_ - delta_y * mouse_sensitivity_, -89.0F, 89.0F);
}

void Camera::set_position(const glm::vec3 position) noexcept {
    position_ = position;
}

glm::mat4 Camera::view_matrix() const noexcept {
    return glm::lookAtRH(position_, position_ + forward(), {0.0F, 1.0F, 0.0F});
}

glm::mat4 Camera::projection_matrix(const float aspect_ratio) const {
    if (aspect_ratio <= 0.0F) {
        throw std::invalid_argument("Camera aspect ratio must be positive");
    }
    glm::mat4 projection = glm::perspectiveRH_ZO(
        field_of_view_radians_, aspect_ratio, near_plane_, far_plane_);
    projection[1][1] *= -1.0F;
    return projection;
}

const glm::vec3& Camera::position() const noexcept {
    return position_;
}

glm::vec3 Camera::forward() const noexcept {
    const float yaw = glm::radians(yaw_degrees_);
    const float pitch = glm::radians(pitch_degrees_);
    return glm::normalize(glm::vec3{
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch),
    });
}

float Camera::field_of_view() const noexcept {
    return field_of_view_radians_;
}

float Camera::near_plane() const noexcept {
    return near_plane_;
}

float Camera::far_plane() const noexcept {
    return far_plane_;
}

}  // namespace vulkancraft::core
