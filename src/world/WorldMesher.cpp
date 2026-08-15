#include "vulkancraft/world/WorldMesher.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace vulkancraft::world {
namespace {

struct FaceDefinition {
    glm::ivec3 neighbor;
    glm::vec3 normal;
    std::uint32_t normal_index;
    std::array<glm::vec3, 4> corners;
};

constexpr std::array<FaceDefinition, 6> faces{{
    {{1, 0, 0}, {1.0F, 0.0F, 0.0F}, 0U, {{{1, 0, 0}, {1, 1, 0}, {1, 1, 1}, {1, 0, 1}}}},
    {{-1, 0, 0}, {-1.0F, 0.0F, 0.0F}, 1U, {{{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 0}}}},
    {{0, 1, 0}, {0.0F, 1.0F, 0.0F}, 2U, {{{0, 1, 1}, {1, 1, 1}, {1, 1, 0}, {0, 1, 0}}}},
    {{0, -1, 0}, {0.0F, -1.0F, 0.0F}, 3U, {{{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}}}},
    {{0, 0, 1}, {0.0F, 0.0F, 1.0F}, 4U, {{{1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 0, 1}}}},
    {{0, 0, -1}, {0.0F, 0.0F, -1.0F}, 5U, {{{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}}},
}};

/** Returns a stable wind class for one connected cloud component. */
[[nodiscard]] std::uint8_t cloud_wind_class(const int global_x, const int global_z) noexcept {
    auto hash = static_cast<std::uint32_t>(global_x) * 0x9E3779B9U;
    hash ^= static_cast<std::uint32_t>(global_z) * 0x85EBCA6BU;
    hash ^= hash >> 16U;
    hash *= 0x7FEB352DU;
    hash ^= hash >> 15U;
    hash *= 0x846CA68BU;
    hash ^= hash >> 16U;
    return static_cast<std::uint8_t>(hash & 0x3U);
}

/** Converts an axis-aligned face normal into the compact shader index. */
[[nodiscard]] std::uint32_t normal_index(const glm::vec3 normal) noexcept {
    if (normal.x > 0.5F) return 0U;
    if (normal.x < -0.5F) return 1U;
    if (normal.y > 0.5F) return 2U;
    if (normal.y < -0.5F) return 3U;
    return normal.z > 0.5F ? 4U : 5U;
}

/** Maps a block to the shader material behavior stored in each vertex. */
[[nodiscard]] VertexMaterial vertex_material_for_block(const BlockType block) noexcept {
    switch (block) {
        case BlockType::leaves:
            return VertexMaterial::leaves;
        case BlockType::water:
            return VertexMaterial::water;
        case BlockType::cloud:
            return VertexMaterial::cloud;
        case BlockType::air:
        case BlockType::grass:
        case BlockType::dirt:
        case BlockType::sand:
        case BlockType::stone:
        case BlockType::log:
            return VertexMaterial::opaque;
    }
    return VertexMaterial::opaque;
}

/** Converts stored Java-style water depth into a visible block-surface height. */
[[nodiscard]] float water_surface_height(
    const std::uint8_t level,
    const bool covered_by_water) noexcept {
    if (covered_by_water || level == falling_water_level) {
        return 1.0F;
    }
    const float horizontal_depth = static_cast<float>(
        std::min(level, maximum_water_flow_level));
    return 0.875F - horizontal_depth * 0.10F;
}

/** Appends one continuous cloud quad to both color and shadow batches. */
void append_cloud_quad(
    WorldMesh& mesh,
    const std::array<glm::vec3, 4>& corners,
    const glm::vec3& normal,
    const std::uint32_t wind_class) {
    constexpr std::array<glm::vec2, 4> texture_coordinates{{
        {96.0F / 128.0F, 96.0F / 96.0F},
        {96.0F / 128.0F, 64.0F / 96.0F},
        {128.0F / 128.0F, 64.0F / 96.0F},
        {128.0F / 128.0F, 96.0F / 96.0F},
    }};
    const auto first_vertex = static_cast<std::uint32_t>(mesh.vertices.size());
    const std::uint32_t attributes = pack_vertex_attributes(
        normal_index(normal), VertexMaterial::cloud, wind_class);
    for (std::size_t corner = 0; corner < corners.size(); ++corner) {
        mesh.vertices.push_back({corners[corner], texture_coordinates[corner], attributes});
    }
    const std::array<std::uint32_t, 6> indices{
        first_vertex,
        first_vertex + 1U,
        first_vertex + 2U,
        first_vertex,
        first_vertex + 2U,
        first_vertex + 3U,
    };
    mesh.cloud_indices.insert(mesh.cloud_indices.end(), indices.begin(), indices.end());
    mesh.shadow_indices.insert(mesh.shadow_indices.end(), indices.begin(), indices.end());
}

/** Emits greedily merged surfaces for one single-height cloud layer. */
void append_cloud_layer(
    const World& world,
    const MeshBuildRegion& region,
    const int cloud_y,
    WorldMesh& mesh) {
    const auto& dimensions = world.dimensions();
    const auto world_index = [&dimensions](const int x, const int z) {
        return static_cast<std::size_t>(z * dimensions.width + x);
    };
    const auto is_cloud = [&world, cloud_y](const int x, const int z) {
        return world.block_at(x, cloud_y, z) == BlockType::cloud;
    };

    const int region_width = region.maximum_x - region.minimum_x;
    const int region_depth = region.maximum_z - region.minimum_z;
    const auto region_index = [region_width, &region](const int x, const int z) {
        return static_cast<std::size_t>(
            (z - region.minimum_z) * region_width + x - region.minimum_x);
    };
    const auto inside_region = [&region](const glm::ivec2 position) {
        return position.x >= region.minimum_x && position.x < region.maximum_x &&
               position.y >= region.minimum_z && position.y < region.maximum_z;
    };

    // Keep only the 32-by-32 output region resident. A sparse flood fill may walk beyond the
    // boundary to preserve one stable wind class for a cloud spanning neighboring chunks.
    std::vector<std::uint8_t> wind_classes(
        static_cast<std::size_t>(region_width * region_depth), 0U);
    std::vector<glm::ivec2> component;
    std::vector<glm::ivec2> pending;
    for (int z = region.minimum_z; z < region.maximum_z; ++z) {
        for (int x = region.minimum_x; x < region.maximum_x; ++x) {
            if (!is_cloud(x, z) || wind_classes[region_index(x, z)] != 0U) {
                continue;
            }
            component.clear();
            pending.clear();
            std::unordered_set<std::size_t> visited;
            pending.push_back({x, z});
            visited.insert(world_index(x, z));
            int minimum_x = x;
            int minimum_z = z;
            while (!pending.empty()) {
                const glm::ivec2 current = pending.back();
                pending.pop_back();
                component.push_back(current);
                minimum_x = std::min(minimum_x, current.x);
                minimum_z = std::min(minimum_z, current.y);
                constexpr std::array<glm::ivec2, 4> neighbors{{
                    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
                }};
                for (const auto neighbor : neighbors) {
                    const int next_x = current.x + neighbor.x;
                    const int next_z = current.y + neighbor.y;
                    if (next_x < 0 || next_x >= dimensions.width || next_z < 0 ||
                        next_z >= dimensions.depth || !is_cloud(next_x, next_z)) {
                        continue;
                    }
                    const std::size_t next = world_index(next_x, next_z);
                    if (visited.insert(next).second) {
                        pending.push_back({next_x, next_z});
                    }
                }
            }
            const std::uint8_t encoded_wind_class = static_cast<std::uint8_t>(
                cloud_wind_class(
                minimum_x + region.world_origin_x,
                minimum_z + region.world_origin_z) + 1U);
            for (const glm::ivec2 position : component) {
                if (inside_region(position)) {
                    wind_classes[region_index(position.x, position.y)] = encoded_wind_class;
                }
            }
        }
    }

    const auto position = [&region, cloud_y](const float x, const float y, const float z) {
        return glm::vec3{
            x + static_cast<float>(region.world_origin_x),
            y + static_cast<float>(cloud_y + region.world_origin_y),
            z + static_cast<float>(region.world_origin_z),
        };
    };
    const auto wind_class_at = [&wind_classes, &region_index](const int x, const int z) {
        return static_cast<std::uint32_t>(wind_classes[region_index(x, z)] - 1U);
    };

    std::vector<bool> merged(
        static_cast<std::size_t>(region_width * region_depth), false);
    for (int z = region.minimum_z; z < region.maximum_z; ++z) {
        for (int x = region.minimum_x; x < region.maximum_x; ++x) {
            if (!is_cloud(x, z) || merged[region_index(x, z)]) {
                continue;
            }
            const std::uint32_t wind_class = wind_class_at(x, z);
            int maximum_x = x + 1;
            while (maximum_x < region.maximum_x && is_cloud(maximum_x, z) &&
                   !merged[region_index(maximum_x, z)] &&
                   wind_class_at(maximum_x, z) == wind_class) {
                ++maximum_x;
            }
            int maximum_z = z + 1;
            for (; maximum_z < region.maximum_z; ++maximum_z) {
                bool complete_row = true;
                for (int candidate_x = x; candidate_x < maximum_x; ++candidate_x) {
                    complete_row = complete_row && is_cloud(candidate_x, maximum_z) &&
                                   !merged[region_index(candidate_x, maximum_z)] &&
                                    wind_class_at(candidate_x, maximum_z) == wind_class;
                }
                if (!complete_row) {
                    break;
                }
            }
            for (int merged_z = z; merged_z < maximum_z; ++merged_z) {
                for (int merged_x = x; merged_x < maximum_x; ++merged_x) {
                    merged[region_index(merged_x, merged_z)] = true;
                }
            }
            if (world.block_at(x, cloud_y + 1, z) != BlockType::cloud) {
                append_cloud_quad(
                    mesh,
                    {position(static_cast<float>(x), 1.0F, static_cast<float>(maximum_z)),
                     position(static_cast<float>(maximum_x), 1.0F, static_cast<float>(maximum_z)),
                     position(static_cast<float>(maximum_x), 1.0F, static_cast<float>(z)),
                     position(static_cast<float>(x), 1.0F, static_cast<float>(z))},
                    {0.0F, 1.0F, 0.0F},
                    wind_class);
            }
            if (world.block_at(x, cloud_y - 1, z) != BlockType::cloud) {
                append_cloud_quad(
                    mesh,
                    {position(static_cast<float>(x), 0.0F, static_cast<float>(z)),
                     position(static_cast<float>(maximum_x), 0.0F, static_cast<float>(z)),
                     position(static_cast<float>(maximum_x), 0.0F, static_cast<float>(maximum_z)),
                     position(static_cast<float>(x), 0.0F, static_cast<float>(maximum_z))},
                    {0.0F, -1.0F, 0.0F},
                    wind_class);
            }
        }
    }

    for (int z = region.minimum_z; z < region.maximum_z; ++z) {
        for (const int direction : {-1, 1}) {
            int x = region.minimum_x;
            while (x < region.maximum_x) {
                if (!is_cloud(x, z) || is_cloud(x, z + direction)) {
                    ++x;
                    continue;
                }
                const int start_x = x;
                const std::uint32_t wind_class = wind_class_at(x, z);
                while (x < region.maximum_x && is_cloud(x, z) &&
                       !is_cloud(x, z + direction) && wind_class_at(x, z) == wind_class) {
                    ++x;
                }
                const float edge_z = static_cast<float>(z + (direction > 0 ? 1 : 0));
                const std::array<glm::vec3, 4> corners = direction > 0
                    ? std::array<glm::vec3, 4>{
                          position(static_cast<float>(x), 0.0F, edge_z),
                          position(static_cast<float>(x), 1.0F, edge_z),
                          position(static_cast<float>(start_x), 1.0F, edge_z),
                          position(static_cast<float>(start_x), 0.0F, edge_z)}
                    : std::array<glm::vec3, 4>{
                          position(static_cast<float>(start_x), 0.0F, edge_z),
                          position(static_cast<float>(start_x), 1.0F, edge_z),
                          position(static_cast<float>(x), 1.0F, edge_z),
                          position(static_cast<float>(x), 0.0F, edge_z)};
                append_cloud_quad(
                    mesh, corners, {0.0F, 0.0F, static_cast<float>(direction)}, wind_class);
            }
        }
    }

    for (int x = region.minimum_x; x < region.maximum_x; ++x) {
        for (const int direction : {-1, 1}) {
            int z = region.minimum_z;
            while (z < region.maximum_z) {
                if (!is_cloud(x, z) || is_cloud(x + direction, z)) {
                    ++z;
                    continue;
                }
                const int start_z = z;
                const std::uint32_t wind_class = wind_class_at(x, z);
                while (z < region.maximum_z && is_cloud(x, z) &&
                       !is_cloud(x + direction, z) && wind_class_at(x, z) == wind_class) {
                    ++z;
                }
                const float edge_x = static_cast<float>(x + (direction > 0 ? 1 : 0));
                const std::array<glm::vec3, 4> corners = direction > 0
                    ? std::array<glm::vec3, 4>{
                          position(edge_x, 0.0F, static_cast<float>(start_z)),
                          position(edge_x, 1.0F, static_cast<float>(start_z)),
                          position(edge_x, 1.0F, static_cast<float>(z)),
                          position(edge_x, 0.0F, static_cast<float>(z))}
                    : std::array<glm::vec3, 4>{
                          position(edge_x, 0.0F, static_cast<float>(z)),
                          position(edge_x, 1.0F, static_cast<float>(z)),
                          position(edge_x, 1.0F, static_cast<float>(start_z)),
                          position(edge_x, 0.0F, static_cast<float>(start_z))};
                append_cloud_quad(
                    mesh, corners, {static_cast<float>(direction), 0.0F, 0.0F}, wind_class);
            }
        }
    }
}

}  // namespace

WorldMesh WorldMesher::build(const World& world) const {
    const auto& dimensions = world.dimensions();
    return build_region(
        world,
        {
            .minimum_x = 0,
            .maximum_x = dimensions.width,
            .minimum_z = 0,
            .maximum_z = dimensions.depth,
        });
}

WorldMesh WorldMesher::build_region(const World& world, const MeshBuildRegion& region) const {
    WorldMesh mesh;
    std::vector<int> cloud_layers;
    const auto& dimensions = world.dimensions();
    if (region.minimum_x < 0 || region.minimum_z < 0 ||
        region.maximum_x > dimensions.width || region.maximum_z > dimensions.depth ||
        region.minimum_x >= region.maximum_x || region.minimum_z >= region.maximum_z) {
        throw std::invalid_argument("Mesh build region is outside the voxel source");
    }

    if (region.surface_lod) {
        struct LodColumn {
            int terrain_height{-1};
            BlockType terrain_block{BlockType::air};
            int water_height{-1};
        };
        struct LodVegetation {
            glm::ivec3 position{};
            BlockType block{BlockType::air};
        };
        const int lod_width = region.maximum_x - region.minimum_x + 2;
        const int lod_depth = region.maximum_z - region.minimum_z + 2;
        const auto lod_index = [lod_width, &region](const int x, const int z) {
            return static_cast<std::size_t>(
                (z - region.minimum_z + 1) * lod_width + x - region.minimum_x + 1);
        };
        std::vector<LodColumn> columns(static_cast<std::size_t>(lod_width * lod_depth));
        std::vector<LodVegetation> vegetation;
        std::vector<bool> lod_cloud_layers(static_cast<std::size_t>(dimensions.height), false);
        const auto is_terrain_material = [](const BlockType block) {
            return block == BlockType::grass || block == BlockType::dirt ||
                   block == BlockType::sand || block == BlockType::stone;
        };
        for (int z = region.minimum_z - 1; z <= region.maximum_z; ++z) {
            for (int x = region.minimum_x - 1; x <= region.maximum_x; ++x) {
                LodColumn& column = columns[lod_index(x, z)];
                for (int y = dimensions.height - 1; y >= 0; --y) {
                    const BlockType block = world.block_at(x, y, z);
                    if (block == BlockType::cloud && x >= region.minimum_x &&
                        x < region.maximum_x && z >= region.minimum_z &&
                        z < region.maximum_z) {
                        lod_cloud_layers[static_cast<std::size_t>(y)] = true;
                    }
                    if ((block == BlockType::log || block == BlockType::leaves) &&
                        x >= region.minimum_x && x < region.maximum_x &&
                        z >= region.minimum_z && z < region.maximum_z) {
                        vegetation.push_back({{x, y, z}, block});
                    }
                    if (column.water_height < 0 && block == BlockType::water) {
                        column.water_height = y;
                    }
                    if (column.terrain_height < 0 && is_terrain_material(block)) {
                        column.terrain_height = y;
                        column.terrain_block = block;
                    }
                }
            }
        }

        const auto append_face = [this, &mesh](
                                     const BlockType block,
                                     const glm::vec3 origin,
                                     const FaceDefinition& face) {
            const auto first_vertex = static_cast<std::uint32_t>(mesh.vertices.size());
            const std::uint32_t attributes = pack_vertex_attributes(
                face.normal_index, vertex_material_for_block(block));
            for (std::size_t corner_index = 0; corner_index < face.corners.size();
                 ++corner_index) {
                glm::vec3 corner = face.corners[corner_index];
                if (block == BlockType::water && corner.y > 0.5F) {
                    corner.y = 0.875F;
                }
                mesh.vertices.push_back({
                    origin + corner,
                    texture_coordinate(block, face.normal, corner_index),
                    attributes});
            }
            auto& target_indices = block == BlockType::water
                                       ? mesh.transparent_indices
                                       : mesh.indices;
            target_indices.insert(
                target_indices.end(),
                {first_vertex,
                 first_vertex + 1U,
                 first_vertex + 2U,
                 first_vertex,
                 first_vertex + 2U,
                 first_vertex + 3U});
            if (block == BlockType::leaves) {
                target_indices.insert(
                    target_indices.end(),
                    {first_vertex,
                     first_vertex + 2U,
                     first_vertex + 1U,
                     first_vertex,
                     first_vertex + 3U,
                     first_vertex + 2U});
            }
        };

        constexpr std::array<std::size_t, 4> side_faces{0U, 1U, 4U, 5U};
        for (int z = region.minimum_z; z < region.maximum_z; ++z) {
            for (int x = region.minimum_x; x < region.maximum_x; ++x) {
                const LodColumn column = columns[lod_index(x, z)];
                if (column.terrain_height >= 0) {
                    append_face(
                        column.terrain_block,
                        {static_cast<float>(x + region.world_origin_x),
                         static_cast<float>(column.terrain_height + region.world_origin_y),
                         static_cast<float>(z + region.world_origin_z)},
                        faces[2]);
                    for (const std::size_t face_index : side_faces) {
                        const FaceDefinition& face = faces[face_index];
                        const bool closes_detail_boundary =
                            (face.neighbor.x < 0 && x == region.minimum_x &&
                             region.close_lod_minimum_x) ||
                            (face.neighbor.x > 0 && x == region.maximum_x - 1 &&
                             region.close_lod_maximum_x) ||
                            (face.neighbor.z < 0 && z == region.minimum_z &&
                             region.close_lod_minimum_z) ||
                            (face.neighbor.z > 0 && z == region.maximum_z - 1 &&
                             region.close_lod_maximum_z);
                        const int neighbor_height = closes_detail_boundary
                                                        ? -1
                                                        : columns[lod_index(
                                                              x + face.neighbor.x,
                                                              z + face.neighbor.z)]
                                                              .terrain_height;
                        for (int y = neighbor_height + 1; y <= column.terrain_height; ++y) {
                            BlockType side_block = world.block_at(x, y, z);
                            if (!is_terrain_material(side_block)) {
                                side_block = BlockType::stone;
                            }
                            append_face(
                                side_block,
                                {static_cast<float>(x + region.world_origin_x),
                                 static_cast<float>(y + region.world_origin_y),
                                 static_cast<float>(z + region.world_origin_z)},
                                face);
                        }
                    }
                }
                if (column.water_height >= 0) {
                    append_face(
                        BlockType::water,
                        {static_cast<float>(x + region.world_origin_x),
                         static_cast<float>(column.water_height + region.world_origin_y),
                         static_cast<float>(z + region.world_origin_z)},
                        faces[2]);
                    for (const std::size_t face_index : side_faces) {
                        const FaceDefinition& face = faces[face_index];
                        const LodColumn neighbor = columns[lod_index(
                            x + face.neighbor.x, z + face.neighbor.z)];
                        const int first_water_y = std::max(
                            {column.terrain_height + 1,
                             neighbor.terrain_height + 1,
                             neighbor.water_height + 1});
                        for (int y = first_water_y; y <= column.water_height; ++y) {
                            append_face(
                                BlockType::water,
                                {static_cast<float>(x + region.world_origin_x),
                                 static_cast<float>(y + region.world_origin_y),
                                 static_cast<float>(z + region.world_origin_z)},
                                face);
                        }
                    }
                }
            }
        }
        for (const LodVegetation& voxel : vegetation) {
            const glm::vec3 origin{
                static_cast<float>(voxel.position.x + region.world_origin_x),
                static_cast<float>(voxel.position.y + region.world_origin_y),
                static_cast<float>(voxel.position.z + region.world_origin_z)};
            for (const FaceDefinition& face : faces) {
                const BlockType neighbor = world.block_at(
                    voxel.position.x + face.neighbor.x,
                    voxel.position.y + face.neighbor.y,
                    voxel.position.z + face.neighbor.z);
                const bool positive_face = face.neighbor.x > 0 || face.neighbor.y > 0 ||
                                           face.neighbor.z > 0;
                const bool visible = voxel.block == BlockType::leaves
                                         ? neighbor == BlockType::air ||
                                               is_transparent(neighbor) ||
                                               (neighbor == BlockType::leaves && positive_face)
                                         : neighbor == BlockType::air ||
                                               is_transparent(neighbor) ||
                                               neighbor == BlockType::leaves;
                if (visible) {
                    append_face(voxel.block, origin, face);
                }
            }
        }
        for (std::size_t y = 0; y < lod_cloud_layers.size(); ++y) {
            if (lod_cloud_layers[y]) {
                append_cloud_layer(world, region, static_cast<int>(y), mesh);
            }
        }
        return mesh;
    }

    const int minimum_section_x = region.minimum_x / World::section_size;
    const int maximum_section_x = (region.maximum_x - 1) / World::section_size;
    const int minimum_section_z = region.minimum_z / World::section_size;
    const int maximum_section_z = (region.maximum_z - 1) / World::section_size;
    for (int section_y = 0; section_y < world.section_count_y(); ++section_y) {
        bool section_range_occupied = false;
        for (int section_z = minimum_section_z;
             section_z <= maximum_section_z && !section_range_occupied;
             ++section_z) {
            for (int section_x = minimum_section_x;
                 section_x <= maximum_section_x && !section_range_occupied;
                 ++section_x) {
                section_range_occupied =
                    world.section_contains_blocks(section_x, section_y, section_z);
            }
        }
        if (!section_range_occupied) {
            continue;
        }
        const int section_begin_y = section_y * World::section_size;
        const int section_end_y = std::min(section_begin_y + World::section_size, dimensions.height);
        for (int y = section_begin_y; y < section_end_y; ++y) {
        for (int z = region.minimum_z; z < region.maximum_z; ++z) {
            for (int x = region.minimum_x; x < region.maximum_x; ++x) {
                const BlockType block = world.block_at(x, y, z);
                if (!is_renderable(block)) {
                    continue;
                }
                if (block == BlockType::cloud) {
                    if (cloud_layers.empty() || cloud_layers.back() != y) {
                        cloud_layers.push_back(y);
                    }
                    continue;
                }

                const glm::vec3 origin{
                    static_cast<float>(x + region.world_origin_x),
                    static_cast<float>(y + region.world_origin_y),
                    static_cast<float>(z + region.world_origin_z)};
                const float current_water_height = block == BlockType::water
                                                       ? water_surface_height(
                                                             world.water_level_at(x, y, z),
                                                             world.block_at(x, y + 1, z) ==
                                                                 BlockType::water)
                                                       : 1.0F;
                for (const auto& face : faces) {
                    if (y == 0 && face.neighbor.y < 0) {
                        continue;
                    }
                    const BlockType neighbor = world.block_at(
                        x + face.neighbor.x,
                        y + face.neighbor.y,
                        z + face.neighbor.z);
                    const bool positive_face = face.neighbor.x > 0 || face.neighbor.y > 0 ||
                                               face.neighbor.z > 0;
                    const bool face_visible = block == BlockType::leaves
                                                  ? neighbor == BlockType::air ||
                                                        is_transparent(neighbor) ||
                                                        (neighbor == BlockType::leaves && positive_face)
                                              : is_transparent(block)
                                                  ? neighbor == BlockType::air ||
                                                        (is_transparent(neighbor) && neighbor != block)
                                                  : neighbor == BlockType::air ||
                                                        is_transparent(neighbor) ||
                                                        neighbor == BlockType::leaves;
                    if (!face_visible) {
                        continue;
                    }

                    const auto first_vertex = static_cast<std::uint32_t>(mesh.vertices.size());
                    const std::uint32_t attributes = pack_vertex_attributes(
                        face.normal_index, vertex_material_for_block(block));
                    for (std::size_t corner_index = 0; corner_index < face.corners.size();
                         ++corner_index) {
                        glm::vec3 corner = face.corners[corner_index];
                        if (block == BlockType::water && corner.y > 0.5F) {
                            corner.y = current_water_height;
                        }
                        mesh.vertices.push_back(
                            {origin + corner,
                             texture_coordinate(
                                 block,
                                 face.normal,
                                 corner_index),
                             attributes});
                    }
                    auto& target_indices = block == BlockType::water
                                               ? mesh.transparent_indices
                                               : mesh.indices;
                    const std::array<std::uint32_t, 6> face_indices{
                        first_vertex,
                        first_vertex + 1U,
                        first_vertex + 2U,
                        first_vertex,
                        first_vertex + 2U,
                        first_vertex + 3U};
                    target_indices.insert(target_indices.end(), face_indices.begin(), face_indices.end());
                    if (block == BlockType::leaves) {
                        target_indices.insert(
                            target_indices.end(),
                            {first_vertex,
                             first_vertex + 2U,
                             first_vertex + 1U,
                             first_vertex,
                             first_vertex + 3U,
                             first_vertex + 2U});
                    }
                }
            }
        }
    }
    }
    for (const int cloud_y : cloud_layers) {
        append_cloud_layer(world, region, cloud_y, mesh);
    }
    return mesh;
}

glm::vec2 WorldMesher::texture_coordinate(
    const BlockType block,
    const glm::vec3& normal,
    const std::size_t corner_index) noexcept {
    struct AtlasRegion {
        float x;
        float y;
        float width{32.0F};
        float height{32.0F};
    };
    constexpr float atlas_width = 128.0F;
    constexpr float atlas_height = 96.0F;
    constexpr std::array<glm::vec2, 4> corners{{
        {0.0F, 1.0F}, {0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}}};

    AtlasRegion region{0.0F, 64.0F};
    switch (block) {
        case BlockType::grass:
            region = normal.y > 0.5F
                         ? AtlasRegion{0.0F, 0.0F}
                         : (normal.y < -0.5F ? AtlasRegion{64.0F, 0.0F}
                                             : AtlasRegion{32.0F, 0.0F});
            break;
        case BlockType::dirt:
            region = {64.0F, 0.0F};
            break;
        case BlockType::sand:
            region = {96.0F, 32.0F};
            break;
        case BlockType::stone:
            region = {0.0F, 64.0F};
            break;
        case BlockType::log:
            region = std::abs(normal.y) > 0.5F ? AtlasRegion{96.0F, 0.0F}
                                               : AtlasRegion{0.0F, 32.0F};
            break;
        case BlockType::leaves:
            region = {64.0F, 32.0F};
            break;
        case BlockType::water:
            region = {32.0F, 64.0F};
            break;
        case BlockType::cloud:
            region = {96.0F, 64.0F};
            break;
        case BlockType::air:
            region = {0.0F, 0.0F};
            break;
    }

    const glm::vec2 local = corners[corner_index];
    return {
        (region.x + local.x * region.width) / atlas_width,
        (region.y + local.y * region.height) / atlas_height,
    };
}

}  // namespace vulkancraft::world
