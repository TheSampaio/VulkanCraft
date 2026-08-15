#pragma once

#include <array>
#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "vulkancraft/core/Camera.hpp"

namespace vulkancraft::rendering {

inline constexpr std::size_t cascade_count = 4;

/** Maximum camera depth covered by dynamic cascaded shadows. */
inline constexpr float maximum_shadow_distance = 320.0F;

/** Contains camera split distances and light transforms for one frame. */
struct CascadeShadowData {
    std::array<glm::mat4, cascade_count> light_view_projections{};
    std::array<float, cascade_count> split_depths{};
};

/** Calculates stable directional-light projections for cascaded shadow maps. */
class CascadeShadowCalculator final {
public:
    /**
     * Calculates all cascades from the current camera frustum.
     *
     * @param camera Camera defining the visible frustum.
     * @param aspect_ratio Positive viewport width divided by height.
     * @param light_direction Normalized direction traveled by sunlight.
     * @param shadow_resolution Width and height of one square cascade layer.
     * @return Light matrices and positive view-space split distances.
     */
    [[nodiscard]] CascadeShadowData calculate(
        const core::Camera& camera,
        float aspect_ratio,
        glm::vec3 light_direction,
        std::uint32_t shadow_resolution) const;

private:
    float split_lambda_{0.92F};
};

}  // namespace vulkancraft::rendering
