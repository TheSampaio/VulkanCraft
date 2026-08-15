#include "vulkancraft/rendering/Frustum.hpp"

#include <utility>

#include <glm/geometric.hpp>

namespace vulkancraft::rendering {
namespace {

/** Returns one row from GLM's column-major matrix representation. */
[[nodiscard]] glm::vec4 matrix_row(const glm::mat4& matrix, const int row) noexcept {
    return {matrix[0][row], matrix[1][row], matrix[2][row], matrix[3][row]};
}

}  // namespace

Frustum::Frustum(std::array<glm::vec4, 6> planes) noexcept : planes_(std::move(planes)) {}

Frustum Frustum::from_clip_matrix(const glm::mat4& clip_from_world) noexcept {
    const glm::vec4 row_x = matrix_row(clip_from_world, 0);
    const glm::vec4 row_y = matrix_row(clip_from_world, 1);
    const glm::vec4 row_z = matrix_row(clip_from_world, 2);
    const glm::vec4 row_w = matrix_row(clip_from_world, 3);
    return Frustum{{
        row_w + row_x,
        row_w - row_x,
        row_w + row_y,
        row_w - row_y,
        row_z,
        row_w - row_z,
    }};
}

bool Frustum::intersects(const AxisAlignedBoundingBox& bounds) const noexcept {
    for (const glm::vec4& plane : planes_) {
        const glm::vec3 positive_vertex{
            plane.x >= 0.0F ? bounds.maximum.x : bounds.minimum.x,
            plane.y >= 0.0F ? bounds.maximum.y : bounds.minimum.y,
            plane.z >= 0.0F ? bounds.maximum.z : bounds.minimum.z,
        };
        if (glm::dot(glm::vec3{plane}, positive_vertex) + plane.w < 0.0F) {
            return false;
        }
    }
    return true;
}

}  // namespace vulkancraft::rendering
