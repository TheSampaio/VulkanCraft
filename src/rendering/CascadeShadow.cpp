#include "vulkancraft/rendering/CascadeShadow.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>

namespace vulkancraft::rendering {
namespace {

[[nodiscard]] glm::mat4 zero_to_one_orthographic(
    const float left,
    const float right,
    const float bottom,
    const float top,
    const float near_depth,
    const float far_depth) {
    glm::mat4 result{1.0F};
    result[0][0] = 2.0F / (right - left);
    result[1][1] = -2.0F / (top - bottom);
    result[2][2] = -1.0F / (far_depth - near_depth);
    result[3][0] = -(right + left) / (right - left);
    result[3][1] = (top + bottom) / (top - bottom);
    result[3][2] = far_depth / (far_depth - near_depth);
    return result;
}

}  // namespace

CascadeShadowData CascadeShadowCalculator::calculate(
    const core::Camera& camera,
    const float aspect_ratio,
    glm::vec3 light_direction,
    const std::uint32_t shadow_resolution) const {
    if (aspect_ratio <= 0.0F || shadow_resolution == 0U || glm::length(light_direction) < 0.001F) {
        throw std::invalid_argument("Invalid cascade shadow calculation input");
    }
    light_direction = glm::normalize(light_direction);

    CascadeShadowData output;
    const float near_plane = camera.near_plane();
    const float far_plane = std::min(camera.far_plane(), maximum_shadow_distance);
    std::array<float, cascade_count + 1> split_distances{};
    split_distances[0] = near_plane;
    for (std::size_t index = 1; index <= cascade_count; ++index) {
        const float ratio = static_cast<float>(index) / static_cast<float>(cascade_count);
        const float logarithmic = near_plane * std::pow(far_plane / near_plane, ratio);
        const float uniform = near_plane + (far_plane - near_plane) * ratio;
        split_distances[index] = std::lerp(uniform, logarithmic, split_lambda_);
        if (index > 0) {
            output.split_depths[index - 1] = split_distances[index];
        }
    }

    const glm::vec3 camera_forward = camera.forward();
    const glm::vec3 camera_right = glm::normalize(
        glm::cross(camera_forward, glm::vec3{0.0F, 1.0F, 0.0F}));
    const glm::vec3 camera_up = glm::normalize(glm::cross(camera_right, camera_forward));
    const float tangent = std::tan(camera.field_of_view() * 0.5F);

    for (std::size_t cascade = 0; cascade < cascade_count; ++cascade) {
        const float slice_near = split_distances[cascade];
        const float slice_far = split_distances[cascade + 1];
        const float near_half_height = tangent * slice_near;
        const float near_half_width = near_half_height * aspect_ratio;
        const float far_half_height = tangent * slice_far;
        const float far_half_width = far_half_height * aspect_ratio;
        const glm::vec3 near_center = camera.position() + camera_forward * slice_near;
        const glm::vec3 far_center = camera.position() + camera_forward * slice_far;

        const std::array<glm::vec3, 8> corners{{
            near_center - camera_right * near_half_width - camera_up * near_half_height,
            near_center + camera_right * near_half_width - camera_up * near_half_height,
            near_center + camera_right * near_half_width + camera_up * near_half_height,
            near_center - camera_right * near_half_width + camera_up * near_half_height,
            far_center - camera_right * far_half_width - camera_up * far_half_height,
            far_center + camera_right * far_half_width - camera_up * far_half_height,
            far_center + camera_right * far_half_width + camera_up * far_half_height,
            far_center - camera_right * far_half_width + camera_up * far_half_height,
        }};

        glm::vec3 center{0.0F};
        for (const glm::vec3& corner : corners) {
            center += corner;
        }
        center /= static_cast<float>(corners.size());

        float radius = 0.0F;
        for (const glm::vec3& corner : corners) {
            radius = std::max(radius, glm::length(corner - center));
        }
        const float guard_band = std::max(8.0F, radius * 0.15F);
        const float extent = std::ceil((radius + guard_band) * 16.0F) / 16.0F;

        const glm::vec3 preferred_up = std::abs(glm::dot(light_direction, glm::vec3{0, 1, 0})) > 0.95F
                                           ? glm::vec3{0, 0, 1}
                                           : glm::vec3{0, 1, 0};
        const glm::mat4 light_view = glm::lookAtRH(
            -light_direction * 100.0F, glm::vec3{0.0F}, preferred_up);

        glm::vec3 minimum{std::numeric_limits<float>::max()};
        glm::vec3 maximum{std::numeric_limits<float>::lowest()};
        for (const glm::vec3& corner : corners) {
            const glm::vec3 light_corner = glm::vec3(light_view * glm::vec4(corner, 1.0F));
            minimum = glm::min(minimum, light_corner);
            maximum = glm::max(maximum, light_corner);
        }

        glm::vec3 light_center = glm::vec3(light_view * glm::vec4(center, 1.0F));
        const float texel_size = extent * 2.0F / static_cast<float>(shadow_resolution);
        light_center.x = std::round(light_center.x / texel_size) * texel_size;
        light_center.y = std::round(light_center.y / texel_size) * texel_size;
        minimum.x = light_center.x - extent;
        maximum.x = light_center.x + extent;
        minimum.y = light_center.y - extent;
        maximum.y = light_center.y + extent;
        minimum.z -= 96.0F;
        maximum.z += 96.0F;

        output.light_view_projections[cascade] = zero_to_one_orthographic(
                                                     minimum.x,
                                                     maximum.x,
                                                     minimum.y,
                                                     maximum.y,
                                                     minimum.z,
                                                     maximum.z) *
                                                 light_view;
    }

    return output;
}

}  // namespace vulkancraft::rendering
