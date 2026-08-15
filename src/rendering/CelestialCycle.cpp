#include "vulkancraft/rendering/CelestialCycle.hpp"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>

namespace vulkancraft::rendering {

float celestial_angle(const float elapsed_seconds) noexcept {
    constexpr float two_pi = 6.28318530718F;
    const float wrapped_seconds = std::fmod(
        std::max(elapsed_seconds, 0.0F), full_day_duration_seconds);
    return wrapped_seconds * two_pi / full_day_duration_seconds;
}

glm::vec3 celestial_direction(const float day_angle) noexcept {
    return {0.0F, std::sin(day_angle), -std::cos(day_angle)};
}

CelestialBasis celestial_basis(const glm::vec3& direction) noexcept {
    constexpr glm::vec3 world_right{1.0F, 0.0F, 0.0F};
    return {
        .right = world_right,
        .up = glm::normalize(glm::cross(world_right, glm::normalize(direction))),
    };
}

}  // namespace vulkancraft::rendering
