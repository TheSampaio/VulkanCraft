#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "vulkancraft/core/Camera.hpp"
#include "vulkancraft/core/PlayerController.hpp"
#include "vulkancraft/rendering/AtlasMipmaps.hpp"
#include "vulkancraft/rendering/CascadeShadow.hpp"
#include "vulkancraft/rendering/CelestialCycle.hpp"
#include "vulkancraft/rendering/Frustum.hpp"
#include "vulkancraft/world/StreamingWorld.hpp"
#include "vulkancraft/world/WaterSimulation.hpp"
#include "vulkancraft/world/WorldGenerator.hpp"
#include "vulkancraft/world/WorldMesher.hpp"

namespace {

int failures = 0;

void expect(const bool condition, const std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

void test_world_bounds() {
    vulkancraft::world::World world{{4, 5, 6}};
    expect(world.size() == 120U, "World allocates every voxel");
    expect(world.block_at(-1, 0, 0) == vulkancraft::world::BlockType::air,
           "Out-of-bounds reads are air");

    world.set_block(2, 3, 4, vulkancraft::world::BlockType::stone);
    expect(world.block_at(2, 3, 4) == vulkancraft::world::BlockType::stone,
           "World stores a block at valid coordinates");
    expect(world.section_contains_blocks(0, 0, 0),
           "Section occupancy records non-air voxel writes");
    world.set_block(2, 3, 4, vulkancraft::world::BlockType::air);
    expect(!world.section_contains_blocks(0, 0, 0),
           "Section occupancy clears after its final block is removed");

    bool rejected = false;
    try {
        world.set_block(4, 0, 0, vulkancraft::world::BlockType::dirt);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    expect(rejected, "World rejects out-of-bounds writes");
}

void test_water_uses_scheduled_java_flow_limits() {
    using vulkancraft::world::BlockType;
    using vulkancraft::world::WaterSimulation;
    using vulkancraft::world::World;

    World flat_world{{21, 4, 21}};
    for (int z = 0; z < 21; ++z) {
        for (int x = 0; x < 21; ++x) {
            flat_world.set_block(x, 0, z, BlockType::stone);
        }
    }
    flat_world.set_water(10, 1, 10, vulkancraft::world::water_source_level);
    WaterSimulation simulation;
    simulation.notify_block_changed(flat_world, {10, 1, 10});
    for (int tick = 0; tick < 8; ++tick) {
        static_cast<void>(simulation.advance(flat_world, WaterSimulation::tick_interval_seconds));
    }
    expect(flat_world.block_at(17, 1, 10) == BlockType::water &&
               flat_world.water_level_at(17, 1, 10) ==
                   WaterSimulation::maximum_horizontal_distance,
           "Water reaches seven horizontal blocks beyond its source");
    expect(flat_world.block_at(18, 1, 10) == BlockType::air,
           "Water does not exceed the Java Edition seven-block horizontal reach");

    World falling_world{{5, 8, 5}};
    for (int z = 0; z < 5; ++z) {
        for (int x = 0; x < 5; ++x) {
            falling_world.set_block(x, 0, z, BlockType::stone);
        }
    }
    falling_world.set_water(2, 6, 2, vulkancraft::world::water_source_level);
    WaterSimulation falling_simulation;
    falling_simulation.notify_block_changed(falling_world, {2, 6, 2});
    static_cast<void>(falling_simulation.advance(
        falling_world, WaterSimulation::tick_interval_seconds));
    expect(falling_world.water_level_at(2, 5, 2) == vulkancraft::world::falling_water_level &&
               falling_world.block_at(2, 4, 2) == BlockType::air,
           "Falling water advances exactly one vertical block per scheduled update");
    static_cast<void>(falling_simulation.advance(
        falling_world, WaterSimulation::tick_interval_seconds));
    expect(falling_world.water_level_at(2, 4, 2) == vulkancraft::world::falling_water_level,
           "Falling water continues downward without a fixed distance limit");
}

void test_default_draw_distance_profile() {
    const vulkancraft::world::WorldStreamingSettings streaming;
    const vulkancraft::core::Camera camera;
    expect(streaming.render_distance == 32,
           "The default draw distance retains exactly 1024 chunks");
    expect(streaming.full_detail_radius == 8,
           "Only the nearby seventeen-by-seventeen window retains full voxels");
    expect(std::abs(camera.far_plane() - 832.0F) < 0.001F,
           "The camera far plane matches the balanced draw-distance profile");
}

void test_generation_is_deterministic() {
    const vulkancraft::world::WorldGenerator generator;
    const vulkancraft::world::WorldGenerationSettings settings{
        .seed = 42U,
        .minimum_y = 0,
        .sea_level = 24,
        .dirt_depth = 4,
        .tree_probability = 0.12F,
    };
    const auto first = generator.generate({32, 72, 32}, settings);
    const auto second = generator.generate({32, 72, 32}, settings);
    const auto different = generator.generate({32, 72, 32}, {
        .seed = 43U,
        .minimum_y = 0,
        .sea_level = 24,
        .tree_probability = 0.12F,
    });

    bool identical = true;
    bool differs_from_other_seed = false;
    for (int y = 0; y < 72; ++y) {
        for (int z = 0; z < 32; ++z) {
            for (int x = 0; x < 32; ++x) {
                identical = identical && first.block_at(x, y, z) == second.block_at(x, y, z);
                differs_from_other_seed = differs_from_other_seed ||
                                          first.block_at(x, y, z) != different.block_at(x, y, z);
            }
        }
    }
    expect(identical, "The same seed produces the same world");
    expect(differs_from_other_seed, "A different seed changes the world");
}

void test_atlas_mipmaps_preserve_sprite_boundaries() {
    constexpr std::uint32_t tile_size = 32U;
    constexpr std::uint32_t width = tile_size * 2U;
    constexpr std::uint32_t height = tile_size;
    std::vector<std::uint8_t> pixels(
        static_cast<std::size_t>(width * height * 4U), 255U);
    for (std::uint32_t y = 0U; y < height; ++y) {
        for (std::uint32_t x = 0U; x < width; ++x) {
            const std::size_t offset = static_cast<std::size_t>(y * width + x) * 4U;
            pixels[offset] = x < tile_size ? 255U : 0U;
            pixels[offset + 1U] = 0U;
            pixels[offset + 2U] = x < tile_size ? 0U : 255U;
        }
    }

    const auto mip_chain = vulkancraft::rendering::build_atlas_mip_chain(
        pixels, width, height, tile_size);
    expect(mip_chain.levels.size() == 6U, "A 32-pixel sprite produces six mip levels");
    const auto& final_level = mip_chain.levels.back();
    expect(final_level.width == 2U && final_level.height == 1U,
           "The final atlas mip retains one pixel per sprite");
    expect(mip_chain.pixels[final_level.byte_offset] == 255U &&
               mip_chain.pixels[final_level.byte_offset + 2U] == 0U,
           "The left sprite stays red through the complete mip chain");
    expect(mip_chain.pixels[final_level.byte_offset + 4U] == 0U &&
               mip_chain.pixels[final_level.byte_offset + 6U] == 255U,
           "The right sprite stays blue without cross-cell bleeding");
}

void test_generation_contains_requested_materials() {
    const vulkancraft::world::WorldGenerator generator;
    const auto generated = generator.generate(
        {96, 96, 96},
        vulkancraft::world::WorldGenerationSettings{
            .seed = 7U,
            .minimum_y = 0,
            .sea_level = 30,
            .dirt_depth = 4,
            .tree_probability = 0.25F,
            .cloud_level = 70,
        });

    std::array<bool, 9> found{};
    int highest_terrain = 0;
    int water_blocks = 0;
    for (int y = 0; y < 96; ++y) {
        for (int z = 0; z < 96; ++z) {
            for (int x = 0; x < 96; ++x) {
                const auto block = generated.block_at(x, y, z);
                found[static_cast<std::size_t>(block)] = true;
                if (block == vulkancraft::world::BlockType::water) {
                    ++water_blocks;
                } else if (block != vulkancraft::world::BlockType::air &&
                           block != vulkancraft::world::BlockType::log &&
                           block != vulkancraft::world::BlockType::leaves) {
                    highest_terrain = std::max(highest_terrain, y);
                }
            }
        }
    }
    for (std::size_t material = 0; material < found.size(); ++material) {
        expect(found[material], "Generated world contains every requested block material");
    }
    expect(water_blocks > 0, "Generated landscape contains ocean or river water");
    expect(highest_terrain >= 50, "Generated landscape contains mountain-scale elevation");
}

void test_subterranean_and_inland_water_features_are_generated() {
    constexpr int width = 288;
    constexpr int height = 160;
    constexpr int depth = 288;
    constexpr int minimum_y = -64;
    constexpr int sea_level = 30;
    const vulkancraft::world::WorldGenerator generator;
    const auto generated = generator.generate(
        {width, height, depth},
        vulkancraft::world::WorldGenerationSettings{
            .seed = 7U,
            .minimum_y = minimum_y,
            .sea_level = sea_level,
            .tree_probability = 0.0F,
            .cloud_level = 1000,
        });

    std::vector<int> solid_heights(static_cast<std::size_t>(width * depth), minimum_y);
    std::size_t inland_water_blocks = 0U;
    std::size_t enclosed_cave_blocks = 0U;
    for (int z = 0; z < depth; ++z) {
        for (int x = 0; x < width; ++x) {
            int highest_solid = minimum_y;
            for (int local_y = 0; local_y < height; ++local_y) {
                const auto block = generated.block_at(x, local_y, z);
                const int global_y = local_y + minimum_y;
                if (block == vulkancraft::world::BlockType::water && global_y > sea_level) {
                    ++inland_water_blocks;
                }
                if (block != vulkancraft::world::BlockType::air &&
                    block != vulkancraft::world::BlockType::water) {
                    highest_solid = global_y;
                }
            }
            solid_heights[static_cast<std::size_t>(z * width + x)] = highest_solid;
            for (int global_y = minimum_y + 4; global_y <= highest_solid - 5; ++global_y) {
                if (generated.block_at(x, global_y - minimum_y, z) ==
                    vulkancraft::world::BlockType::air) {
                    ++enclosed_cave_blocks;
                }
            }
        }
    }

    std::size_t ravine_columns = 0U;
    for (int z = 1; z < depth - 1; ++z) {
        for (int x = 1; x < width - 1; ++x) {
            const int center = solid_heights[static_cast<std::size_t>(z * width + x)];
            const int neighboring_peak = std::max({
                solid_heights[static_cast<std::size_t>(z * width + x - 1)],
                solid_heights[static_cast<std::size_t>(z * width + x + 1)],
                solid_heights[static_cast<std::size_t>((z - 1) * width + x)],
                solid_heights[static_cast<std::size_t>((z + 1) * width + x)],
            });
            if (neighboring_peak > sea_level + 9 && neighboring_peak - center >= 12) {
                ++ravine_columns;
            }
        }
    }

    if (inland_water_blocks <= 100U) {
        std::cerr << "Observed elevated inland water blocks: " << inland_water_blocks << '\n';
    }
    expect(inland_water_blocks > 100U, "Terrain generation creates elevated inland lakes");
    expect(enclosed_cave_blocks > 1'000U, "Terrain generation creates underground cave tunnels");
    expect(ravine_columns > 20U, "Terrain generation creates deep open ravines");
}

void test_cloud_shapes_vary_and_stay_two_blocks_high() {
    constexpr int cloud_level = 70;
    constexpr int cell_size = 40;
    const vulkancraft::world::WorldGenerator generator;
    const auto generated = generator.generate(
        {cell_size * 3, 80, cell_size * 3},
        vulkancraft::world::WorldGenerationSettings{
            .seed = 73U,
            .minimum_y = 0,
            .sea_level = 30,
            .tree_probability = 0.0F,
            .cloud_level = cloud_level,
        });

    std::array<int, 9> cell_cloud_counts{};
    for (int z = 0; z < cell_size * 3; ++z) {
        for (int x = 0; x < cell_size * 3; ++x) {
            if (generated.block_at(x, cloud_level, z) ==
                vulkancraft::world::BlockType::cloud) {
                ++cell_cloud_counts[static_cast<std::size_t>(
                    (z / cell_size) * 3 + x / cell_size)];
            }
            expect(generated.block_at(x, cloud_level - 1, z) !=
                       vulkancraft::world::BlockType::cloud &&
                       generated.block_at(x, cloud_level + 1, z) ==
                           generated.block_at(x, cloud_level, z) &&
                       generated.block_at(x, cloud_level + 2, z) !=
                           vulkancraft::world::BlockType::cloud,
                   "Clouds occupy exactly two matching voxel layers");
        }
    }
    std::set<int> distinct_occupied_areas;
    int total_cloud_blocks = 0;
    for (const int count : cell_cloud_counts) {
        if (count > 0) {
            distinct_occupied_areas.insert(count);
            total_cloud_blocks += count;
        }
    }
    expect(distinct_occupied_areas.size() >= 3U,
           "Cloud cells contain multiple deterministic rectangle silhouettes");
    expect(total_cloud_blocks >= cell_size * cell_size,
           "Cloud coverage remains dense across the generated region");
}

void test_mesher_culls_shared_faces() {
    const vulkancraft::world::WorldMesher mesher;
    vulkancraft::world::World single{{1, 1, 1}};
    single.set_block(0, 0, 0, vulkancraft::world::BlockType::stone);
    const auto single_mesh = mesher.build(single);
    expect(single_mesh.vertices.size() == 20U, "The world floor omits its unreachable bottom face");
    expect(single_mesh.indices.size() == 30U, "One floor cube has ten visible triangles");

    vulkancraft::world::World pair{{2, 1, 1}};
    pair.set_block(0, 0, 0, vulkancraft::world::BlockType::stone);
    pair.set_block(1, 0, 0, vulkancraft::world::BlockType::stone);
    const auto pair_mesh = mesher.build(pair);
    expect(pair_mesh.vertices.size() == 32U, "Two floor cubes omit bottom and shared faces");
    expect(pair_mesh.indices.size() == 48U, "Two touching floor cubes emit eight quads");

    for (const auto& vertex : pair_mesh.vertices) {
        expect(vertex.texture_coordinate.x >= 0.0F && vertex.texture_coordinate.x <= 1.0F &&
                   vertex.texture_coordinate.y >= 0.0F && vertex.texture_coordinate.y <= 1.0F,
               "Generated atlas coordinates remain normalized");
    }
}

void test_distant_surface_lod_removes_hidden_detail() {
    vulkancraft::world::World world{{4, 8, 4}};
    for (int z = 0; z < 4; ++z) {
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 3; ++y) {
                world.set_block(
                    x,
                    y,
                    z,
                    y == 2 ? vulkancraft::world::BlockType::grass
                           : vulkancraft::world::BlockType::stone);
            }
        }
    }
    world.set_block(1, 3, 1, vulkancraft::world::BlockType::log);
    world.set_block(1, 4, 1, vulkancraft::world::BlockType::leaves);

    const vulkancraft::world::WorldMesher mesher;
    const auto full = mesher.build(world);
    const auto distant = mesher.build_region(
        world,
        {.minimum_x = 0,
         .maximum_x = 4,
         .minimum_z = 0,
         .maximum_z = 4,
         .surface_lod = true});
    expect(!distant.vertices.empty() && distant.indices.size() < full.indices.size() * 2U,
           "Distant heightfield LOD keeps a bounded surface representation");
    expect(std::any_of(
               distant.vertices.begin(),
               distant.vertices.end(),
               [](const vulkancraft::world::Vertex& vertex) {
                   return vulkancraft::world::vertex_material(vertex.attributes) ==
                          vulkancraft::world::VertexMaterial::leaves;
               }),
           "Distant heightfield LOD preserves cutout tree foliage");
    const auto open_boundary = mesher.build_region(
        world,
        {.minimum_x = 1,
         .maximum_x = 4,
         .minimum_z = 0,
         .maximum_z = 4,
         .surface_lod = true});
    const auto sealed_distant = mesher.build_region(
        world,
        {.minimum_x = 1,
         .maximum_x = 4,
         .minimum_z = 0,
         .maximum_z = 4,
         .surface_lod = true,
         .close_lod_minimum_x = true});
    expect(sealed_distant.indices.size() > open_boundary.indices.size(),
           "The full-detail-to-heightfield boundary is sealed below terrain");
}

void test_distant_ocean_lod_keeps_the_seabed() {
    vulkancraft::world::World ocean{{4, 8, 4}};
    for (int z = 0; z < 4; ++z) {
        for (int x = 0; x < 4; ++x) {
            ocean.set_block(x, 0, z, vulkancraft::world::BlockType::sand);
            for (int y = 1; y <= 3; ++y) {
                ocean.set_block(x, y, z, vulkancraft::world::BlockType::water);
            }
        }
    }

    const vulkancraft::world::WorldMesher mesher;
    const auto distant = mesher.build_region(
        ocean,
        {.minimum_x = 0,
         .maximum_x = 4,
         .minimum_z = 0,
         .maximum_z = 4,
         .surface_lod = true});

    expect(!distant.indices.empty(),
           "Distant ocean LOD retains an opaque seabed below translucent water");
    expect(!distant.transparent_indices.empty(),
           "Distant ocean LOD retains the translucent water surface above its seabed");
    const bool has_sand = std::any_of(
        distant.vertices.begin(), distant.vertices.end(), [](const auto& vertex) {
            return vulkancraft::world::vertex_material(vertex.attributes) ==
                       vulkancraft::world::VertexMaterial::opaque &&
                   std::abs(vertex.position.y - 1.0F) < 0.0001F;
        });
    expect(has_sand, "The distant ocean seabed closes the view toward the world void");
}

void test_atlas_tiles_and_orientation_are_stable() {
    const vulkancraft::world::WorldMesher mesher;
    const auto verify_tile = [&mesher](
                                 const vulkancraft::world::BlockType block,
                                 const float expected_x,
                                 const float expected_y,
                                 const std::string_view message) {
        vulkancraft::world::World world{{2, 2, 1}};
        world.set_block(0, 0, 0, block);
        world.set_block(1, 0, 0, block);
        const auto mesh = mesher.build(world);
        float minimum_x = 1.0F;
        float maximum_x = 0.0F;
        float minimum_y = 1.0F;
        float maximum_y = 0.0F;
        for (const auto& vertex : mesh.vertices) {
            minimum_x = std::min(minimum_x, vertex.texture_coordinate.x);
            maximum_x = std::max(maximum_x, vertex.texture_coordinate.x);
            minimum_y = std::min(minimum_y, vertex.texture_coordinate.y);
            maximum_y = std::max(maximum_y, vertex.texture_coordinate.y);
        }
        expect(std::abs(minimum_x - expected_x / 128.0F) < 0.0001F &&
                   std::abs(maximum_x - (expected_x + 32.0F) / 128.0F) < 0.0001F &&
                   std::abs(minimum_y - expected_y / 96.0F) < 0.0001F &&
                   std::abs(maximum_y - (expected_y + 32.0F) / 96.0F) < 0.0001F,
               message);

        expect(mesh.vertices.size() >= 32U &&
                   mesh.vertices[0].texture_coordinate == mesh.vertices[16].texture_coordinate,
               "Equal blocks retain the same atlas orientation");
    };
    verify_tile(
        vulkancraft::world::BlockType::leaves,
        64.0F,
        32.0F,
        "Leaves use the bright foliage tile in the compact atlas");
    verify_tile(
        vulkancraft::world::BlockType::water,
        32.0F,
        64.0F,
        "Water uses the first blue tile in the compact atlas");

    vulkancraft::world::World log_world{{1, 3, 1}};
    log_world.set_block(0, 1, 0, vulkancraft::world::BlockType::log);
    const auto log_mesh = mesher.build(log_world);
    for (const auto& vertex : log_mesh.vertices) {
        const std::uint32_t normal = vulkancraft::world::vertex_normal_index(vertex.attributes);
        const bool end_face = normal == 2U || normal == 3U;
        const float expected_minimum_x = end_face ? 96.0F : 0.0F;
        const float expected_minimum_y = end_face ? 0.0F : 32.0F;
        expect(vertex.texture_coordinate.x >= expected_minimum_x / 128.0F &&
                   vertex.texture_coordinate.x <= (expected_minimum_x + 32.0F) / 128.0F &&
                   vertex.texture_coordinate.y >= expected_minimum_y / 96.0F &&
                   vertex.texture_coordinate.y <= (expected_minimum_y + 32.0F) / 96.0F,
               end_face ? "Log top and bottom faces use the compact log-end tile"
                        : "Log side faces use the compact bark tile");
    }

    vulkancraft::world::World leaves_world{{1, 1, 1}};
    leaves_world.set_block(0, 0, 0, vulkancraft::world::BlockType::leaves);
    const auto leaves_mesh = mesher.build(leaves_world);
    expect(leaves_mesh.indices.size() == 60U,
           "Leaves duplicate winding so every exposed face renders from inside");
    for (std::size_t face = 1; face < leaves_mesh.vertices.size() / 4U; ++face) {
        for (std::size_t corner = 0; corner < 4U; ++corner) {
            expect(
                leaves_mesh.vertices[face * 4U + corner].texture_coordinate ==
                    leaves_mesh.vertices[corner].texture_coordinate,
                "Every leaf face retains the same fixed texture orientation");
        }
    }

    vulkancraft::world::World dense_leaves{{2, 2, 1}};
    dense_leaves.set_block(0, 1, 0, vulkancraft::world::BlockType::leaves);
    dense_leaves.set_block(1, 1, 0, vulkancraft::world::BlockType::leaves);
    const auto dense_leaves_mesh = mesher.build(dense_leaves);
    expect(dense_leaves_mesh.vertices.size() == 44U &&
               dense_leaves_mesh.indices.size() == 132U,
           "Adjacent leaves retain one double-sided internal density plane");

    vulkancraft::world::World trunk_in_leaves{{2, 3, 1}};
    trunk_in_leaves.set_block(0, 1, 0, vulkancraft::world::BlockType::log);
    trunk_in_leaves.set_block(1, 1, 0, vulkancraft::world::BlockType::leaves);
    const auto trunk_in_leaves_mesh = mesher.build(trunk_in_leaves);
    const auto visible_log_side_vertices = std::count_if(
        trunk_in_leaves_mesh.vertices.begin(),
        trunk_in_leaves_mesh.vertices.end(),
        [](const vulkancraft::world::Vertex& vertex) {
            const std::uint32_t normal =
                vulkancraft::world::vertex_normal_index(vertex.attributes);
            return vulkancraft::world::vertex_material(vertex.attributes) ==
                       vulkancraft::world::VertexMaterial::opaque &&
                   normal != 2U && normal != 3U &&
                   vertex.texture_coordinate.y >= 32.0F / 96.0F &&
                   vertex.texture_coordinate.y <= 64.0F / 96.0F;
        });
    expect(visible_log_side_vertices == 16,
           "Every log side remains rendered when the neighboring block is cutout leaves");

    vulkancraft::world::World water_world{{1, 2, 1}};
    water_world.set_block(0, 0, 0, vulkancraft::world::BlockType::water);
    const auto water_mesh = mesher.build(water_world);
    float maximum_water_y = 0.0F;
    for (const auto& vertex : water_mesh.vertices) {
        maximum_water_y = std::max(maximum_water_y, vertex.position.y);
    }
    expect(std::abs(maximum_water_y - 0.875F) < 0.0001F,
           "Water surfaces sit below the full block height");

    vulkancraft::world::World shallow_flow_world{{1, 2, 1}};
    shallow_flow_world.set_water(0, 0, 0, vulkancraft::world::maximum_water_flow_level);
    const auto shallow_flow_mesh = mesher.build(shallow_flow_world);
    float maximum_shallow_water_y = 0.0F;
    for (const auto& vertex : shallow_flow_mesh.vertices) {
        maximum_shallow_water_y = std::max(maximum_shallow_water_y, vertex.position.y);
    }
    expect(std::abs(maximum_shallow_water_y - 0.175F) < 0.0001F,
           "Horizontal water depth lowers the rendered surface without scaling its atlas cell");

    vulkancraft::world::World cloud_world{{1, 2, 1}};
    cloud_world.set_block(0, 1, 0, vulkancraft::world::BlockType::cloud);
    const auto cloud_mesh = mesher.build(cloud_world);
    expect(cloud_mesh.cloud_indices.size() == 36U &&
               cloud_mesh.transparent_indices.empty(),
           "Cloud color geometry uses a batch separate from water");
    expect(!cloud_mesh.vertices.empty() &&
               vulkancraft::world::vertex_material(
                   cloud_mesh.vertices.front().attributes) ==
                   vulkancraft::world::VertexMaterial::cloud,
           "Cloud geometry retains its compact material classification");
    expect(cloud_mesh.shadow_indices.size() == 36U,
           "Cloud geometry is included in the dedicated shadow-caster batch");
    expect(vulkancraft::world::vertex_animation_speed(
               cloud_mesh.vertices.front().attributes) >= 0.055F &&
               vulkancraft::world::vertex_animation_speed(
                   cloud_mesh.vertices.front().attributes) <= 0.13F,
           "Cloud geometry receives a slow deterministic wind speed");

    vulkancraft::world::World flat_cloud_world{{8, 2, 8}};
    for (int z = 0; z < 8; ++z) {
        for (int x = 0; x < 8; ++x) {
            flat_cloud_world.set_block(x, 1, z, vulkancraft::world::BlockType::cloud);
        }
    }
    const auto flat_cloud_mesh = mesher.build(flat_cloud_world);
    expect(flat_cloud_mesh.vertices.size() == 24U &&
               flat_cloud_mesh.cloud_indices.size() == 36U,
           "A flat cloud merges into six continuous faces without block-grid seams");
    expect(flat_cloud_mesh.shadow_indices.size() == 36U,
           "Cloud shadow geometry also excludes every internal voxel face");
    expect(std::all_of(
                flat_cloud_mesh.vertices.begin(),
                flat_cloud_mesh.vertices.end(),
                [&flat_cloud_mesh](const vulkancraft::world::Vertex& vertex) {
                    return vulkancraft::world::vertex_wind_class(vertex.attributes) ==
                           vulkancraft::world::vertex_wind_class(
                               flat_cloud_mesh.vertices.front().attributes);
                }),
           "Every face in one connected cloud moves at the same speed");

    vulkancraft::world::World stacked_cloud_world{{8, 3, 8}};
    for (int y = 1; y <= 2; ++y) {
        for (int z = 0; z < 8; ++z) {
            for (int x = 0; x < 8; ++x) {
                stacked_cloud_world.set_block(x, y, z, vulkancraft::world::BlockType::cloud);
            }
        }
    }
    const auto stacked_cloud_mesh = mesher.build(stacked_cloud_world);
    expect(stacked_cloud_mesh.cloud_indices.size() == 60U,
           "Two-block-high clouds omit the shared horizontal interior surface");

    vulkancraft::world::World separated_clouds{{16, 2, 16}};
    separated_clouds.set_block(1, 1, 1, vulkancraft::world::BlockType::cloud);
    separated_clouds.set_block(14, 1, 14, vulkancraft::world::BlockType::cloud);
    const auto separated_cloud_mesh = mesher.build(separated_clouds);
    const float first_cloud_speed = vulkancraft::world::vertex_animation_speed(
        separated_cloud_mesh.vertices[0].attributes);
    const float second_cloud_speed = vulkancraft::world::vertex_animation_speed(
        separated_cloud_mesh.vertices[24].attributes);
    const int first_wind_layer = static_cast<int>(std::lround(
        (first_cloud_speed - 0.055F) / 0.025F));
    const int second_wind_layer = static_cast<int>(std::lround(
        (second_cloud_speed - 0.055F) / 0.025F));
    expect(separated_cloud_mesh.vertices.size() == 48U &&
               first_cloud_speed != second_cloud_speed,
           "Separated clouds receive natural deterministic speed variation");
    expect(first_wind_layer != second_wind_layer &&
               std::abs(first_wind_layer - second_wind_layer) * 2.5F > 2.0F,
           "Clouds with different speeds occupy nonintersecting altitude layers");
}

void test_player_jumps_and_swims() {
    vulkancraft::world::StreamingWorld streaming{
        {.seed = 17U, .minimum_y = -64, .sea_level = 62, .tree_probability = 0.0F},
        {.chunk_size = 32,
         .render_distance = 3,
         .full_detail_radius = 1,
         .world_height = 384,
         .minimum_y = -64,
         .generation_padding = 3}};
    static_cast<void>(streaming.initialize({1.5F, 90.0F, 1.5F}));
    vulkancraft::core::PlayerController player;
    player.spawn(streaming, 1.5F, 1.5F);
    const int spawn_x = static_cast<int>(std::floor(player.position().x));
    const int spawn_z = static_cast<int>(std::floor(player.position().z));
    expect(streaming.block_at(spawn_x, static_cast<int>(std::floor(player.position().y)), spawn_z) !=
               vulkancraft::world::BlockType::water,
           "Player spawning searches for nearby dry terrain");
    for (int step = 0; step < 20; ++step) {
        player.update({}, {0.0F, 0.0F, -1.0F}, 0.016F, streaming);
    }
    const float standing_y = player.position().y;
    player.update({.jump = true}, {0.0F, 0.0F, -1.0F}, 0.016F, streaming);
    player.update({}, {0.0F, 0.0F, -1.0F}, 0.10F, streaming);
    expect(player.position().y > standing_y + 0.2F, "Space makes a grounded player jump");

    player.set_position({5.5F, 100.1F, 5.5F});
    for (int y = 94; y <= 106; ++y) {
        static_cast<void>(streaming.set_block(5, y, 5, vulkancraft::world::BlockType::water));
    }
    const float water_start_y = player.position().y;
    for (int step = 0; step < 10; ++step) {
        player.update({}, {0.0F, 0.0F, -1.0F}, 0.05F, streaming);
    }
    expect(player.is_swimming(), "Water around the body enables swimming physics");
    expect(player.position().y > water_start_y - 1.0F, "A submerged player sinks slowly");
    const float submerged_y = player.position().y;
    for (int step = 0; step < 10; ++step) {
        player.update({.jump = true}, {0.0F, 0.0F, -1.0F}, 0.05F, streaming);
    }
    expect(player.position().y > submerged_y, "Holding Space makes a submerged player rise");

    player.update({}, {0.0F, 0.0F, -1.0F}, 0.02F, streaming);
    player.update({.jump = true}, {0.0F, 0.0F, -1.0F}, 0.02F, streaming);
    player.update({}, {0.0F, 0.0F, -1.0F}, 0.05F, streaming);
    player.update({.jump = true}, {0.0F, 0.0F, -1.0F}, 0.02F, streaming);
    expect(player.is_flying(), "Two quick Space presses enable flight mode");
    const float flight_start_y = player.position().y;
    player.update({}, {0.0F, 0.0F, -1.0F}, 0.05F, streaming);
    expect(std::abs(player.position().y - flight_start_y) < 0.0001F,
           "A flying player remains vertically static without ascent or descent input");
    player.update({.jump = true}, {0.0F, 0.0F, -1.0F}, 0.05F, streaming);
    expect(player.position().y > flight_start_y, "Space moves a flying player upward");

    player.update({}, {0.0F, 0.0F, -1.0F}, 0.02F, streaming);
    player.update({.jump = true}, {0.0F, 0.0F, -1.0F}, 0.02F, streaming);
    player.update({}, {0.0F, 0.0F, -1.0F}, 0.05F, streaming);
    player.update({.jump = true}, {0.0F, 0.0F, -1.0F}, 0.02F, streaming);
    expect(!player.is_flying(), "A second Space double-tap disables flight mode");
}

void test_global_regions_share_boundaries() {
    const vulkancraft::world::WorldGenerator generator;
    const vulkancraft::world::WorldGenerationSettings settings{
        .seed = 8128U,
        .minimum_y = 0,
        .sea_level = 24,
        .tree_probability = 0.0F,
        .cloud_level = 1000,
    };
    const auto left = generator.generate_region({17, 72, 17}, 0, -8, settings);
    const auto right = generator.generate_region({17, 72, 17}, 16, -8, settings);
    for (int y = 0; y < 72; ++y) {
        for (int z = 0; z < 17; ++z) {
            expect(left.block_at(16, y, z) == right.block_at(0, y, z),
                   "Adjacent generated regions agree at global boundaries");
        }
    }
}

void test_surface_only_generation_matches_visible_columns() {
    const vulkancraft::world::WorldGenerator generator;
    const vulkancraft::world::WorldGenerationSettings settings{
        .seed = 0x51FACEU,
        .minimum_y = -64,
        .sea_level = 62,
        .tree_probability = 0.025F,
        .cloud_level = 136,
    };
    constexpr vulkancraft::world::WorldDimensions dimensions{48, 224, 48};
    const auto full = generator.generate_region(dimensions, -173, 91, settings, false, false);
    const auto compact = generator.generate_region(
        dimensions, -173, 91, settings, false, false, true);
    const auto is_terrain = [](const vulkancraft::world::BlockType block) {
        return block == vulkancraft::world::BlockType::grass ||
               block == vulkancraft::world::BlockType::dirt ||
               block == vulkancraft::world::BlockType::sand ||
               block == vulkancraft::world::BlockType::stone;
    };
    const auto highest = [&](const vulkancraft::world::World& world,
                             const int x,
                             const int z,
                             const auto predicate) {
        for (int y = dimensions.height - 1; y >= 0; --y) {
            if (predicate(world.block_at(x, y, z))) {
                return y;
            }
        }
        return -1;
    };
    std::size_t full_terrain_voxels = 0U;
    std::size_t compact_terrain_voxels = 0U;
    for (int z = 0; z < dimensions.depth; ++z) {
        for (int x = 0; x < dimensions.width; ++x) {
            expect(highest(full, x, z, is_terrain) == highest(compact, x, z, is_terrain),
                   "Surface-only generation preserves visible terrain height");
            const auto is_water = [](const vulkancraft::world::BlockType block) {
                return block == vulkancraft::world::BlockType::water;
            };
            expect(highest(full, x, z, is_water) == highest(compact, x, z, is_water),
                   "Surface-only generation preserves visible water height");
            for (int y = 0; y < dimensions.height; ++y) {
                full_terrain_voxels += is_terrain(full.block_at(x, y, z)) ? 1U : 0U;
                compact_terrain_voxels += is_terrain(compact.block_at(x, y, z)) ? 1U : 0U;
            }
        }
    }
    expect(compact_terrain_voxels * 12U < full_terrain_voxels,
           "Surface-only generation omits hidden underground voxel writes");
}

void test_streaming_crosses_positive_and_negative_chunks() {
    vulkancraft::world::StreamingWorld streaming{
        {.seed = 91U,
         .minimum_y = -64,
         .sea_level = 62,
         .tree_probability = 0.02F,
         .cloud_level = 192},
        {.chunk_size = 32,
         .render_distance = 3,
         .full_detail_radius = 1,
         .world_height = 384,
         .minimum_y = -64,
         .generation_padding = 3}};
    const auto initial = streaming.initialize({1.0F, 40.0F, 1.0F});
    expect(!initial.vertices.empty(), "Streaming creates an initial render mesh");
    expect(streaming.center_chunk() == vulkancraft::world::ChunkCoordinate{0, 0},
           "Initial stream center uses the camera chunk");
    expect(streaming.set_block(2, 200, 2, vulkancraft::world::BlockType::water),
           "A loaded source-water block can be scheduled");
    streaming.advance_water(vulkancraft::world::WaterSimulation::tick_interval_seconds);
    expect(streaming.water_level_at(2, 199, 2) == vulkancraft::world::falling_water_level,
           "Streaming water advances downward through the active voxel window");
    expect(!streaming.update({31.9F, 80.0F, 31.9F}).has_value(),
           "Movement inside one chunk does not rebuild geometry");
    expect(streaming.set_block(-1, 100, -33, vulkancraft::world::BlockType::stone),
           "A loaded global block can be edited");
    const auto dirty_chunks = streaming.consume_dirty_chunk_indices();
    expect(!dirty_chunks.empty() &&
               !streaming.chunk_mesh(dirty_chunks.front()).indices.empty(),
           "A block edit publishes only affected cached chunks");
    expect(!streaming.update({1.0F, 80.0F, 1.0F}).has_value(),
           "A local block edit does not rebuild the aggregate streaming mesh");
    auto negative = streaming.update({-0.1F, 80.0F, -32.1F});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!negative.has_value() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
        negative = streaming.update({-0.1F, 80.0F, -32.1F});
    }
    expect(negative.has_value() && !streaming.chunk_meshes().empty(),
           "Crossing into negative chunks publishes cached chunk geometry");
    expect(negative.has_value() && negative->mesh.draw_ranges.size() == 7U &&
               negative->replacement_indices.size() == 7U,
           "Streaming packs the seven entering chunks in one shifted 3 by 3 update");
    expect(streaming.center_chunk() == vulkancraft::world::ChunkCoordinate{-1, -2},
           "Negative positions use floor-based chunk coordinates");
    expect(streaming.block_at(-1, 100, -33) == vulkancraft::world::BlockType::stone,
           "Block edits survive an asynchronous window replacement");
    expect(streaming.water_level_at(2, 199, 2) == vulkancraft::world::falling_water_level,
           "Flowing-water metadata survives an asynchronous window replacement");

    for (int step = 0; step < 12; ++step) {
        const glm::vec3 target{
            static_cast<float>((step + 1) * 32) + 0.25F,
            100.0F,
            -63.75F};
        auto streamed = streaming.update(target);
        const auto step_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!streamed.has_value() && std::chrono::steady_clock::now() < step_deadline) {
            std::this_thread::yield();
            streamed = streaming.update(target);
        }
        expect(streamed.has_value(), "Repeated long-distance streaming completes without failure");
    }
    expect(streaming.center_chunk() == vulkancraft::world::ChunkCoordinate{12, -2},
           "Streaming remains stable after traveling many chunks from spawn");
}

void test_cascade_splits_and_matrices() {
    const vulkancraft::core::Camera camera;
    const vulkancraft::rendering::CascadeShadowCalculator calculator;
    const auto cascades = calculator.calculate(camera, 16.0F / 9.0F, {0.4F, -0.8F, 0.2F}, 2048U);

    float previous = camera.near_plane();
    for (const float split : cascades.split_depths) {
        expect(split > previous, "Cascade split distances increase");
        expect(split <= vulkancraft::rendering::maximum_shadow_distance + 0.001F,
               "Cascade split stays inside the configured shadow distance");
        previous = split;
    }
    expect(std::abs(cascades.split_depths.back() -
                    vulkancraft::rendering::maximum_shadow_distance) < 0.001F,
           "Last cascade ends at the bounded dynamic-shadow distance");

    for (const auto& matrix : cascades.light_view_projections) {
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row) {
                expect(std::isfinite(matrix[column][row]), "Cascade matrix contains finite values");
            }
        }
    }

    const glm::vec3 light_direction = glm::normalize(glm::vec3{0.4F, -0.8F, 0.2F});
    const glm::vec3 reference = camera.position() + camera.forward() * 10.0F;
    const glm::vec4 toward_light = cascades.light_view_projections[0] *
                                   glm::vec4(reference - light_direction * 3.0F, 1.0F);
    const glm::vec4 away_from_light = cascades.light_view_projections[0] *
                                      glm::vec4(reference + light_direction * 3.0F, 1.0F);
    expect(toward_light.z / toward_light.w < away_from_light.z / away_from_light.w,
           "Cascade depth increases away from the directional light");

    const auto expect_caster_guard_band = [&](const vulkancraft::core::Camera& guarded_camera) {
        constexpr float aspect = 16.0F / 9.0F;
        const auto guarded = calculator.calculate(
            guarded_camera, aspect, {0.4F, -0.8F, 0.2F}, 2048U);
        const float slice_near = guarded_camera.near_plane();
        const float slice_far = guarded.split_depths.front();
        const glm::vec3 forward = guarded_camera.forward();
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3{0.0F, 1.0F, 0.0F}));
        const glm::vec3 up = glm::normalize(glm::cross(right, forward));
        const float half_height = std::tan(guarded_camera.field_of_view() * 0.5F) * slice_far;
        const float half_width = half_height * aspect;
        const glm::vec3 slice_center = guarded_camera.position() +
                                       forward * ((slice_near + slice_far) * 0.5F);
        const glm::vec3 far_corner = guarded_camera.position() + forward * slice_far +
                                     right * half_width + up * half_height;
        const glm::vec3 guarded_caster =
            far_corner + glm::normalize(far_corner - slice_center) * 4.0F;
        const glm::vec4 clip = guarded.light_view_projections.front() *
                               glm::vec4(guarded_caster, 1.0F);
        expect(std::abs(clip.x / clip.w) < 1.0F && std::abs(clip.y / clip.w) < 1.0F,
               "Cascade guard band retains off-frustum casters while the camera moves");
    };
    expect_caster_guard_band(camera);
    vulkancraft::core::Camera rotated_camera = camera;
    rotated_camera.look(730.0F, -120.0F);
    expect_caster_guard_band(rotated_camera);
}

void test_vulkan_frustum_culls_axis_aligned_bounds() {
    using vulkancraft::rendering::AxisAlignedBoundingBox;
    using vulkancraft::rendering::Frustum;

    const Frustum identity = Frustum::from_clip_matrix(glm::mat4{1.0F});
    expect(identity.intersects({{-.5F, -.5F, 0.1F}, {.5F, .5F, .9F}}),
           "A box inside the Vulkan clip volume remains visible");
    expect(identity.intersects({{-.5F, -.5F, -0.1F}, {.5F, .5F, .1F}}),
           "A box crossing the near plane remains conservatively visible");
    expect(!identity.intersects({{-3.0F, -.5F, .1F}, {-2.0F, .5F, .9F}}),
           "A box beyond the left plane is culled");
    expect(!identity.intersects({{2.0F, -.5F, .1F}, {3.0F, .5F, .9F}}),
           "A box beyond the right plane is culled");
    expect(!identity.intersects({{-.5F, -.5F, -2.0F}, {.5F, .5F, -1.0F}}),
           "A box behind the Vulkan near plane is culled");
    expect(!identity.intersects({{-.5F, -.5F, 2.0F}, {.5F, .5F, 3.0F}}),
           "A box beyond the far plane is culled");

    const vulkancraft::core::Camera camera;
    const glm::mat4 camera_clip = camera.projection_matrix(16.0F / 9.0F) * camera.view_matrix();
    const Frustum camera_frustum = Frustum::from_clip_matrix(camera_clip);
    const glm::vec3 forward_center = camera.position() + camera.forward() * 40.0F;
    const glm::vec3 behind_center = camera.position() - camera.forward() * 20.0F;
    expect(camera_frustum.intersects({forward_center - glm::vec3{2.0F},
                                      forward_center + glm::vec3{2.0F}}),
           "A world box in front of the camera remains visible");
    expect(!camera_frustum.intersects({behind_center - glm::vec3{2.0F},
                                       behind_center + glm::vec3{2.0F}}),
           "A world box behind the camera is culled");
}

void test_celestial_cycle_is_centered() {
    constexpr float half_pi = 1.57079632679F;
    constexpr float pi = 3.14159265359F;
    const glm::vec3 sunrise = vulkancraft::rendering::celestial_direction(0.0F);
    const glm::vec3 noon = vulkancraft::rendering::celestial_direction(half_pi);
    const glm::vec3 sunset = vulkancraft::rendering::celestial_direction(pi);

    expect(std::abs(vulkancraft::rendering::celestial_angle(720.0F) - pi) < 0.0001F &&
               std::abs(vulkancraft::rendering::celestial_angle(1440.0F)) < 0.0001F,
           "Day and night each last twelve real-time minutes");

    expect(std::abs(sunrise.x) < 0.0001F && std::abs(sunrise.y) < 0.0001F &&
               std::abs(sunrise.z + 1.0F) < 0.0001F,
           "Sunrise begins at the center of the default forward horizon");
    expect(std::abs(noon.x) < 0.0001F && std::abs(noon.y - 1.0F) < 0.0001F &&
               std::abs(noon.z) < 0.0001F,
           "The celestial orbit crosses directly overhead without lateral drift");
    expect(std::abs(sunset.x) < 0.0001F && std::abs(sunset.y) < 0.0001F &&
               std::abs(sunset.z - 1.0F) < 0.0001F,
           "Sunset ends at the center of the opposite horizon");

    for (const glm::vec3 direction : {sunrise, noon, sunset}) {
        const auto basis = vulkancraft::rendering::celestial_basis(direction);
        expect(glm::length(basis.right - glm::vec3{1.0F, 0.0F, 0.0F}) < 0.0001F,
               "Celestial horizontal orientation remains fixed to the world X axis");
        expect(std::abs(glm::dot(basis.right, direction)) < 0.0001F &&
                   std::abs(glm::dot(basis.up, direction)) < 0.0001F &&
                   std::abs(glm::dot(basis.right, basis.up)) < 0.0001F,
               "Celestial world axes remain orthogonal throughout the orbit");
    }
}

}  // namespace

int main() {
    test_world_bounds();
    test_water_uses_scheduled_java_flow_limits();
    test_default_draw_distance_profile();
    test_generation_is_deterministic();
    test_atlas_mipmaps_preserve_sprite_boundaries();
    test_generation_contains_requested_materials();
    test_subterranean_and_inland_water_features_are_generated();
    test_cloud_shapes_vary_and_stay_two_blocks_high();
    test_mesher_culls_shared_faces();
    test_distant_surface_lod_removes_hidden_detail();
    test_distant_ocean_lod_keeps_the_seabed();
    test_atlas_tiles_and_orientation_are_stable();
    test_global_regions_share_boundaries();
    test_surface_only_generation_matches_visible_columns();
    test_streaming_crosses_positive_and_negative_chunks();
    test_player_jumps_and_swims();
    test_cascade_splits_and_matrices();
    test_vulkan_frustum_culls_axis_aligned_bounds();
    test_celestial_cycle_is_centered();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "All VulkanCraft tests passed\n";
    return 0;
}
