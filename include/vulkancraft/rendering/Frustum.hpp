#pragma once

#include <array>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace vulkancraft::rendering {

/** Defines a world-space axis-aligned bounding box. */
struct AxisAlignedBoundingBox {
    glm::vec3 minimum{};
    glm::vec3 maximum{};
};

/** Stores the six world-space planes of a Vulkan clip volume. */
class Frustum final {
public:
    /**
     * Extracts a frustum from a world-to-clip transform.
     *
     * The near plane follows Vulkan's zero-to-one clip-space depth convention.
     *
     * @param clip_from_world Homogeneous world-to-clip transformation.
     * @return Frustum whose planes face the visible volume.
     */
    [[nodiscard]] static Frustum from_clip_matrix(const glm::mat4& clip_from_world) noexcept;

    /**
     * Conservatively tests whether an axis-aligned box intersects the frustum.
     *
     * @param bounds World-space box with ordered minimum and maximum corners.
     * @return False only when the complete box lies outside at least one plane.
     */
    [[nodiscard]] bool intersects(const AxisAlignedBoundingBox& bounds) const noexcept;

private:
    explicit Frustum(std::array<glm::vec4, 6> planes) noexcept;

    std::array<glm::vec4, 6> planes_;
};

}  // namespace vulkancraft::rendering
