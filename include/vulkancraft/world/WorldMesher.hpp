#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "vulkancraft/world/World.hpp"

namespace vulkancraft::world {

/** Identifies shader behavior without storing per-vertex display colors. */
enum class VertexMaterial : std::uint32_t {
    opaque = 0,
    leaves = 1,
    water = 2,
    cloud = 3,
};

/** First speed used by deterministic cloud wind classes. */
inline constexpr float minimum_cloud_speed = 0.055F;

/** Difference between consecutive deterministic cloud wind classes. */
inline constexpr float cloud_speed_step = 0.025F;

/** Number of discrete wind classes stored in two packed bits. */
inline constexpr std::uint32_t cloud_speed_class_count = 4U;

/**
 * Packs a face normal, material, and cloud wind class into one 32-bit value.
 *
 * @param normal_index Axis-aligned normal index in the inclusive range zero through five.
 * @param material Shader material classification.
 * @param wind_class Cloud wind class in the inclusive range zero through three.
 * @return Compact attributes consumed as an unsigned integer vertex input.
 */
[[nodiscard]] constexpr std::uint32_t pack_vertex_attributes(
    const std::uint32_t normal_index,
    const VertexMaterial material,
    const std::uint32_t wind_class = 0U) noexcept {
    return (normal_index & 0x7U) | (static_cast<std::uint32_t>(material) << 3U) |
           ((wind_class & 0x3U) << 5U);
}

/** Returns the axis-aligned normal index stored in compact vertex attributes. */
[[nodiscard]] constexpr std::uint32_t vertex_normal_index(
    const std::uint32_t attributes) noexcept {
    return attributes & 0x7U;
}

/** Returns the material stored in compact vertex attributes. */
[[nodiscard]] constexpr VertexMaterial vertex_material(const std::uint32_t attributes) noexcept {
    return static_cast<VertexMaterial>((attributes >> 3U) & 0x3U);
}

/** Returns the cloud wind class stored in compact vertex attributes. */
[[nodiscard]] constexpr std::uint32_t vertex_wind_class(
    const std::uint32_t attributes) noexcept {
    return (attributes >> 5U) & 0x3U;
}

/** Returns the decoded deterministic cloud animation speed. */
[[nodiscard]] constexpr float vertex_animation_speed(const std::uint32_t attributes) noexcept {
    return vertex_material(attributes) == VertexMaterial::cloud
               ? minimum_cloud_speed +
                     static_cast<float>(vertex_wind_class(attributes)) * cloud_speed_step
               : 0.0F;
}

/** Describes one world-space vertex consumed by the Vulkan pipelines. */
struct Vertex {
    glm::vec3 position{};
    glm::vec2 texture_coordinate{};
    std::uint32_t attributes{};
};

static_assert(sizeof(Vertex) == 24U, "The streamed vertex format must remain compact");

/** Locates one cached chunk inside the combined renderer buffers. */
struct MeshDrawRange {
    std::uint32_t first_index{};
    std::uint32_t index_count{};
    std::uint32_t first_transparent_index{};
    std::uint32_t transparent_index_count{};
    std::uint32_t first_cloud_index{};
    std::uint32_t cloud_index_count{};
    std::uint32_t first_shadow_index{};
    std::uint32_t shadow_index_count{};
};

/** Owns an indexed triangle mesh generated from a voxel world. */
struct WorldMesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> transparent_indices;
    std::vector<std::uint32_t> cloud_indices;
    std::vector<std::uint32_t> shadow_indices;
    std::vector<MeshDrawRange> draw_ranges;
};

/** Selects a local subregion and positions it in global world space. */
struct MeshBuildRegion {
    int minimum_x{};
    int maximum_x{};
    int minimum_z{};
    int maximum_z{};
    int world_origin_x{};
    int world_origin_y{};
    int world_origin_z{};
    bool surface_lod{false};
    bool close_lod_minimum_x{false};
    bool close_lod_maximum_x{false};
    bool close_lod_minimum_z{false};
    bool close_lod_maximum_z{false};
};

/** Converts solid voxels into a face-culled render mesh. */
class WorldMesher final {
public:
    /**
     * Emits one quad for each solid-to-air boundary.
     *
     * @param world Immutable voxel source.
     * @return Indexed triangle mesh in world coordinates.
     */
    [[nodiscard]] WorldMesh build(const World& world) const;

    /**
     * Emits a cropped local region at a global horizontal origin.
     *
     * @param world Immutable voxel source, including any generation padding.
     * @param region Half-open local bounds and global source origin.
     * @return Indexed triangle mesh using global x and z positions.
     */
    [[nodiscard]] WorldMesh build_region(const World& world, const MeshBuildRegion& region) const;

private:
    /** Returns one normalized atlas coordinate for a face corner. */
    [[nodiscard]] static glm::vec2 texture_coordinate(
        BlockType block,
        const glm::vec3& normal,
        std::size_t corner_index) noexcept;
};

}  // namespace vulkancraft::world
