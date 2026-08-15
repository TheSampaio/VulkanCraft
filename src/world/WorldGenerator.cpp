#include "vulkancraft/world/WorldGenerator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <optional>

#include "vulkancraft/core/Parallel.hpp"

namespace vulkancraft::world {
namespace {

[[nodiscard]] std::uint32_t mix_bits(std::uint32_t value) noexcept {
    value ^= value >> 16U;
    value *= 0x7FEB352DU;
    value ^= value >> 15U;
    value *= 0x846CA68BU;
    value ^= value >> 16U;
    return value;
}

[[nodiscard]] float smooth_step(const float value) noexcept {
    return value * value * (3.0F - 2.0F * value);
}

[[nodiscard]] float value_noise(const float x, const float z, const std::uint32_t seed) noexcept {
    const auto x0 = static_cast<int>(std::floor(x));
    const auto z0 = static_cast<int>(std::floor(z));
    const float tx = smooth_step(x - static_cast<float>(x0));
    const float tz = smooth_step(z - static_cast<float>(z0));

    const auto sample = [seed](const int sample_x, const int sample_z) {
        const auto ux = std::bit_cast<std::uint32_t>(sample_x);
        const auto uz = std::bit_cast<std::uint32_t>(sample_z);
        const std::uint32_t hash = mix_bits(ux * 0x9E3779B9U ^ uz * 0x85EBCA6BU ^ seed);
        return static_cast<float>(hash & 0x00FFFFFFU) / static_cast<float>(0x00FFFFFFU);
    };

    const float bottom = std::lerp(sample(x0, z0), sample(x0 + 1, z0), tx);
    const float top = std::lerp(sample(x0, z0 + 1), sample(x0 + 1, z0 + 1), tx);
    return std::lerp(bottom, top, tz) * 2.0F - 1.0F;
}

[[nodiscard]] int floor_divide(const int value, const int divisor) noexcept {
    const int quotient = value / divisor;
    const int remainder = value % divisor;
    return remainder < 0 ? quotient - 1 : quotient;
}

[[nodiscard]] float random_unit_value(
    const int x,
    const int z,
    const std::uint32_t seed) noexcept {
    const auto ux = std::bit_cast<std::uint32_t>(x);
    const auto uz = std::bit_cast<std::uint32_t>(z);
    const std::uint32_t hash = mix_bits(ux * 0x27D4EB2DU ^ uz * 0x165667B1U ^ seed);
    return static_cast<float>(hash & 0x00FFFFFFU) / static_cast<float>(0x00FFFFFFU);
}

struct LakeSample {
    float terrain_height{};
    int water_level{};
};

/** Lowers suitable lowland terrain into sparse deterministic lake basins. */
[[nodiscard]] LakeSample sample_lake(
    const int global_x,
    const int global_z,
    const float original_height,
    const float continental,
    const float mountain_mask,
    const WorldGenerationSettings& settings) noexcept {
    LakeSample result{original_height, settings.sea_level};
    if (original_height > static_cast<float>(settings.sea_level + 22) ||
        continental < -0.12F || mountain_mask > 0.20F) {
        return result;
    }

    constexpr int cell_size = 128;
    const int cell_x = floor_divide(global_x, cell_size);
    const int cell_z = floor_divide(global_z, cell_size);
    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            const int candidate_x = cell_x + offset_x;
            const int candidate_z = cell_z + offset_z;
            if (random_unit_value(
                    candidate_x, candidate_z, settings.seed ^ 0xD1B54A35U) < 0.50F) {
                continue;
            }

            const int anchor_x = candidate_x * cell_size + 20 + static_cast<int>(
                random_unit_value(candidate_x, candidate_z, settings.seed ^ 0x94D049BBU) *
                88.0F);
            const int anchor_z = candidate_z * cell_size + 20 + static_cast<int>(
                random_unit_value(candidate_x, candidate_z, settings.seed ^ 0x369DEA0FU) *
                88.0F);
            const float radius_x = 13.0F + random_unit_value(
                                                  candidate_x,
                                                  candidate_z,
                                                  settings.seed ^ 0xDB4F0B91U) *
                                                  13.0F;
            const float radius_z = 11.0F + random_unit_value(
                                                 candidate_x,
                                                 candidate_z,
                                                 settings.seed ^ 0xBBE05633U) *
                                                 11.0F;
            const float normalized_x = static_cast<float>(global_x - anchor_x) / radius_x;
            const float normalized_z = static_cast<float>(global_z - anchor_z) / radius_z;
            const float distance_squared = normalized_x * normalized_x +
                                           normalized_z * normalized_z;
            if (distance_squared >= 1.0F) {
                continue;
            }

            const float strength = smooth_step(1.0F - distance_squared);
            const int lake_level = settings.sea_level + 2 + static_cast<int>(
                random_unit_value(candidate_x, candidate_z, settings.seed ^ 0xA0F2EC75U) *
                4.0F);
            const float lake_bed = static_cast<float>(lake_level - 3) - strength * 4.0F;
            const float carved_height = std::lerp(original_height, lake_bed, strength);
            if (carved_height < result.terrain_height) {
                result.terrain_height = carved_height;
                if (carved_height < static_cast<float>(lake_level)) {
                    result.water_level = std::max(result.water_level, lake_level);
                }
            }
        }
    }
    return result;
}

/** Returns the bottom of a sparse open ravine, or no value outside a ravine. */
[[nodiscard]] std::optional<int> sample_ravine_bottom(
    const int global_x,
    const int global_z,
    const int surface_height,
    const float mountain_mask,
    const WorldGenerationSettings& settings) noexcept {
    if (surface_height <= settings.sea_level + 9 || mountain_mask > 0.9F) {
        return std::nullopt;
    }

    struct Direction {
        float x;
        float z;
    };
    constexpr std::array directions{
        Direction{1.0F, 0.0F},
        Direction{0.92388F, 0.38268F},
        Direction{0.70711F, 0.70711F},
        Direction{0.38268F, 0.92388F},
        Direction{0.0F, 1.0F},
        Direction{-0.38268F, 0.92388F},
        Direction{-0.70711F, 0.70711F},
        Direction{-0.92388F, 0.38268F},
    };

    constexpr int cell_size = 224;
    const int cell_x = floor_divide(global_x, cell_size);
    const int cell_z = floor_divide(global_z, cell_size);
    float strongest = 0.0F;
    int selected_depth = 0;
    for (int offset_z = -1; offset_z <= 1; ++offset_z) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            const int candidate_x = cell_x + offset_x;
            const int candidate_z = cell_z + offset_z;
            if (random_unit_value(
                    candidate_x, candidate_z, settings.seed ^ 0xC13FA9A9U) < 0.80F) {
                continue;
            }
            const float anchor_x = static_cast<float>(candidate_x * cell_size) +
                                   random_unit_value(
                                       candidate_x,
                                       candidate_z,
                                       settings.seed ^ 0x91E10DA5U) *
                                       static_cast<float>(cell_size);
            const float anchor_z = static_cast<float>(candidate_z * cell_size) +
                                   random_unit_value(
                                       candidate_x,
                                       candidate_z,
                                       settings.seed ^ 0xF1357AE9U) *
                                       static_cast<float>(cell_size);
            const auto direction_index = static_cast<std::size_t>(
                random_unit_value(candidate_x, candidate_z, settings.seed ^ 0xA24BAED4U) *
                static_cast<float>(directions.size())) % directions.size();
            const Direction direction = directions[direction_index];
            const float relative_x = static_cast<float>(global_x) - anchor_x;
            const float relative_z = static_cast<float>(global_z) - anchor_z;
            const float along = relative_x * direction.x + relative_z * direction.z;
            const float across = std::abs(-relative_x * direction.z + relative_z * direction.x);
            const float half_length = 45.0F + random_unit_value(
                                                    candidate_x,
                                                    candidate_z,
                                                    settings.seed ^ 0xB7E15162U) *
                                                    32.0F;
            const float half_width = 3.0F + random_unit_value(
                                                   candidate_x,
                                                   candidate_z,
                                                   settings.seed ^ 0x8AED2A6BU) *
                                                   4.0F;
            if (std::abs(along) >= half_length || across >= half_width) {
                continue;
            }
            const float width_strength = 1.0F - across / half_width;
            const float end_strength = 1.0F - std::pow(std::abs(along) / half_length, 6.0F);
            const float strength = width_strength * end_strength;
            if (strength > strongest) {
                strongest = strength;
                selected_depth = 22 + static_cast<int>(random_unit_value(
                    candidate_x, candidate_z, settings.seed ^ 0x4CF5AD43U) * 34.0F);
            }
        }
    }

    if (strongest < 0.18F) {
        return std::nullopt;
    }
    const int depth = std::max(4, static_cast<int>(static_cast<float>(selected_depth) *
                                                   std::sqrt(strongest)));
    return std::max(settings.minimum_y + 4, surface_height - depth);
}

/** Carves two inexpensive, continuous tunnel families below one terrain column. */
void carve_caves(
    World& world,
    const int local_x,
    const int local_z,
    const int global_x,
    const int global_z,
    const int surface_height,
    const WorldGenerationSettings& settings) {
    constexpr int lane_size = 72;
    const auto carve_lane = [&](const bool runs_along_x) {
        const int along = runs_along_x ? global_x : global_z;
        const int across = runs_along_x ? global_z : global_x;
        const int lane = floor_divide(across, lane_size);
        const std::uint32_t orientation_salt = runs_along_x ? 0x6A09E667U : 0xBB67AE85U;
        if (random_unit_value(lane, 0, settings.seed ^ orientation_salt) < 0.24F) {
            return;
        }

        const float lane_center = static_cast<float>(lane * lane_size + 15) +
                                  random_unit_value(
                                      lane, 1, settings.seed ^ orientation_salt ^ 0x3C6EF372U) *
                                      42.0F;
        const float bend = value_noise(
            static_cast<float>(along) * 0.014F,
            static_cast<float>(lane) * 0.37F,
            settings.seed ^ orientation_salt ^ 0xA54FF53AU) * 7.0F;
        const float radius_noise = value_noise(
            static_cast<float>(along) * 0.009F,
            static_cast<float>(lane) * 0.61F,
            settings.seed ^ orientation_salt ^ 0x510E527FU);
        const float radius = 2.2F + random_unit_value(
                                         lane,
                                         2,
                                         settings.seed ^ orientation_salt ^ 0x9B05688CU) *
                                         1.8F +
                             std::max(0.0F, radius_noise) * 2.4F;
        const float horizontal_distance = std::abs(static_cast<float>(across) - lane_center - bend);
        if (horizontal_distance >= radius) {
            return;
        }

        const float center_y = static_cast<float>(settings.sea_level - 22) +
                               random_unit_value(
                                   lane, 3, settings.seed ^ orientation_salt ^ 0x1F83D9ABU) *
                                   34.0F +
                               value_noise(
                                   static_cast<float>(along) * 0.008F,
                                   static_cast<float>(lane) * 0.43F,
                                   settings.seed ^ orientation_salt ^ 0x5BE0CD19U) * 9.0F;
        const float vertical_radius =
            std::sqrt(radius * radius - horizontal_distance * horizontal_distance) * 0.78F;
        const int minimum = std::max(
            settings.minimum_y + 4, static_cast<int>(std::floor(center_y - vertical_radius)));
        const int maximum = std::min(
            surface_height - 5, static_cast<int>(std::ceil(center_y + vertical_radius)));
        for (int global_y = minimum; global_y <= maximum; ++global_y) {
            world.set_block_untracked(
                local_x, global_y - settings.minimum_y, local_z, BlockType::air);
        }
    };

    carve_lane(true);
    carve_lane(false);
}

}  // namespace

World WorldGenerator::generate(
    const WorldDimensions dimensions,
    const WorldGenerationSettings& settings) const {
    return generate_region(dimensions, 0, 0, settings);
}

World WorldGenerator::generate_region(
    const WorldDimensions dimensions,
    const int origin_x,
    const int origin_z,
    const WorldGenerationSettings& settings,
    const bool build_section_index,
    const bool use_worker_pool,
    const bool surface_only) const {
    if (dimensions.width < 1 || dimensions.height < 24 || dimensions.depth < 1) {
        throw std::invalid_argument("Generated worlds require a height of at least 24 voxels");
    }
    const int maximum_y = settings.minimum_y + dimensions.height;
    if (settings.dirt_depth < 1 || settings.sea_level < settings.minimum_y + 6 ||
        settings.sea_level >= maximum_y - 10 || settings.tree_probability < 0.0F ||
        settings.tree_probability > 1.0F) {
        throw std::invalid_argument("Invalid world generation settings");
    }

    World world{dimensions};
    std::vector<int> surface_heights(
        static_cast<std::size_t>(dimensions.width * dimensions.depth), 0);

    const auto generate_terrain_row = [&](const std::size_t row) {
        const int z = static_cast<int>(row);
        for (int x = 0; x < dimensions.width; ++x) {
            const int global_x = origin_x + x;
            const int global_z = origin_z + z;
            const float sample_x = static_cast<float>(global_x);
            const float sample_z = static_cast<float>(global_z);
            const float continental = fractal_noise(
                sample_x * 0.0024F, sample_z * 0.0024F, settings.seed ^ 0xA341316CU);
            const float hills = fractal_noise(
                sample_x * 0.012F, sample_z * 0.012F, settings.seed ^ 0xC8013EA4U);
            const float ridge_noise = fractal_noise(
                sample_x * 0.0055F, sample_z * 0.0055F, settings.seed ^ 0xAD90777DU);
            const float ridge = 1.0F - std::abs(ridge_noise);
            const float mountain_mask = std::clamp((continental - 0.08F) * 2.4F, 0.0F, 1.0F);
            const float mountains = std::pow(ridge, 3.0F) * mountain_mask * 52.0F;

            float terrain_height = static_cast<float>(settings.sea_level + 5) + continental * 28.0F +
                                   hills * 7.0F + mountains;
            if (continental < -0.10F) {
                terrain_height = static_cast<float>(settings.sea_level - 5) +
                                 continental * 17.0F + hills * 3.0F;
            }

            const float river_distance = std::abs(fractal_noise(
                sample_x * 0.0038F, sample_z * 0.0038F, settings.seed ^ 0x7E95761EU));
            const bool river_lowland = terrain_height <= static_cast<float>(settings.sea_level + 18) &&
                                       mountain_mask < 0.14F;
            if (continental > -0.16F && river_lowland && river_distance < 0.045F) {
                const float river_strength = 1.0F - river_distance / 0.045F;
                const float river_bed = static_cast<float>(settings.sea_level - 2) -
                                        river_strength * 5.0F;
                terrain_height = std::min(terrain_height, river_bed);
            }

            const LakeSample lake = sample_lake(
                global_x,
                global_z,
                terrain_height,
                continental,
                mountain_mask,
                settings);
            terrain_height = lake.terrain_height;

            const int global_surface = std::clamp(
                static_cast<int>(std::round(terrain_height)),
                settings.minimum_y + 2,
                maximum_y - 8);
            const int surface = global_surface - settings.minimum_y;
            const auto surface_index = static_cast<std::size_t>(z * dimensions.width + x);
            surface_heights[surface_index] = surface;

            const bool sandy = global_surface <= settings.sea_level + 1;
            if (surface_only) {
                int visible_surface = surface;
                BlockType visible_block = sandy ? BlockType::sand : BlockType::grass;
                if (lake.water_level == settings.sea_level) {
                    const std::optional<int> ravine_bottom = sample_ravine_bottom(
                        global_x, global_z, global_surface, mountain_mask, settings);
                    if (ravine_bottom.has_value()) {
                        visible_surface = *ravine_bottom - 1 - settings.minimum_y;
                        visible_block = BlockType::stone;
                        surface_heights[surface_index] = visible_surface;
                    }
                }
                world.set_block_untracked(x, visible_surface, z, visible_block);
                for (int global_y = global_surface + 1; global_y <= lake.water_level; ++global_y) {
                    world.set_block_untracked(
                        x, global_y - settings.minimum_y, z, BlockType::water);
                }
                continue;
            }
            for (int y = 0; y <= surface; ++y) {
                BlockType block = BlockType::stone;
                const int depth_below_surface = surface - y;
                if (depth_below_surface == 0) {
                    block = sandy ? BlockType::sand : BlockType::grass;
                } else if (depth_below_surface < settings.dirt_depth) {
                    block = sandy ? BlockType::sand : BlockType::dirt;
                }
                world.set_block_untracked(x, y, z, block);
            }
            for (int global_y = global_surface + 1; global_y <= lake.water_level; ++global_y) {
                world.set_block_untracked(
                    x, global_y - settings.minimum_y, z, BlockType::water);
            }

            carve_caves(world, x, z, global_x, global_z, global_surface, settings);
            if (lake.water_level == settings.sea_level) {
                const std::optional<int> ravine_bottom = sample_ravine_bottom(
                    global_x, global_z, global_surface, mountain_mask, settings);
                if (ravine_bottom.has_value()) {
                    for (int global_y = *ravine_bottom; global_y <= global_surface; ++global_y) {
                        world.set_block_untracked(
                            x, global_y - settings.minimum_y, z, BlockType::air);
                    }
                }
            }
        }
    };
    if (use_worker_pool) {
        core::parallel_for(static_cast<std::size_t>(dimensions.depth), generate_terrain_row);
    } else {
        for (int row = 0; row < dimensions.depth; ++row) {
            generate_terrain_row(static_cast<std::size_t>(row));
        }
    }

    for (int z = 3; z < dimensions.depth - 3; ++z) {
        for (int x = 3; x < dimensions.width - 3; ++x) {
            const auto surface_index = static_cast<std::size_t>(z * dimensions.width + x);
            const int surface = surface_heights[surface_index];
            const int global_surface = surface + settings.minimum_y;
            if (global_surface <= settings.sea_level + 1 ||
                world.block_at(x, surface, z) != BlockType::grass) {
                continue;
            }

            const int global_x = origin_x + x;
            const int global_z = origin_z + z;
            const float tree_roll = random_unit(
                global_x, global_z, settings.seed ^ 0xA511E9B3U);
            if (tree_roll >= settings.tree_probability) {
                continue;
            }

            const int trunk_height = 4 + static_cast<int>(
                                             random_unit(
                                                 global_x,
                                                 global_z,
                                                 settings.seed ^ 0x63D83595U) *
                                             3.0F);
            place_tree(world, x, surface, z, trunk_height);
        }
    }

    const int local_cloud_y = settings.cloud_level - settings.minimum_y;
    if (local_cloud_y >= 0 && local_cloud_y < dimensions.height) {
        constexpr int cloud_scale = 3;
        constexpr int cloud_cell_size = 40 * cloud_scale;
        const auto generate_cloud_row = [&](const std::size_t row) {
            const int z = static_cast<int>(row);
            for (int x = 0; x < dimensions.width; ++x) {
                const int global_x = origin_x + x;
                const int global_z = origin_z + z;
                const int cell_x = floor_divide(global_x, cloud_cell_size);
                const int cell_z = floor_divide(global_z, cloud_cell_size);
                bool inside_cloud = false;
                for (int neighbor_z = -1; neighbor_z <= 1 && !inside_cloud; ++neighbor_z) {
                    for (int neighbor_x = -1; neighbor_x <= 1 && !inside_cloud; ++neighbor_x) {
                        const int candidate_x = cell_x + neighbor_x;
                        const int candidate_z = cell_z + neighbor_z;
                        const int cell_origin_x = candidate_x * cloud_cell_size;
                        const int cell_origin_z = candidate_z * cloud_cell_size;
                        for (int cluster = 0; cluster < 2 && !inside_cloud; ++cluster) {
                            const std::uint32_t cluster_salt =
                                static_cast<std::uint32_t>(cluster) * 0x9E3779B9U;
                            const float presence = random_unit(
                                candidate_x,
                                candidate_z,
                                settings.seed ^ 0x46B7D91FU ^ cluster_salt);
                            if (presence < (cluster == 0 ? 0.04F : 0.22F)) {
                                continue;
                            }
                            const int anchor_x = cell_origin_x + 4 * cloud_scale + static_cast<int>(random_unit(
                                 candidate_x,
                                 candidate_z,
                                 settings.seed ^ 0xD3A2646CU ^ cluster_salt) *
                                 static_cast<float>(24 * cloud_scale));
                            const int anchor_z = cell_origin_z + 4 * cloud_scale + static_cast<int>(random_unit(
                                 candidate_x,
                                 candidate_z,
                                 settings.seed ^ 0x8F4C21D7U ^ cluster_salt) *
                                 static_cast<float>(24 * cloud_scale));
                            const int orientation = static_cast<int>(random_unit(
                                candidate_x,
                                candidate_z,
                                settings.seed ^ 0xFE32A187U ^ cluster_salt) * 4.0F) % 4;
                            const int relative_x = global_x - anchor_x;
                            const int relative_z = global_z - anchor_z;
                            const int along = orientation == 0 ? relative_x
                                              : orientation == 1 ? relative_z
                                              : orientation == 2 ? -relative_x
                                                                 : -relative_z;
                            const int across = orientation == 0 ? relative_z
                                               : orientation == 1 ? -relative_x
                                               : orientation == 2 ? -relative_z
                                                                  : relative_x;
                            const int length = cloud_scale * (8 + static_cast<int>(random_unit(
                                 candidate_x,
                                 candidate_z,
                                 settings.seed ^ 0x56B4E2A1U ^ cluster_salt) * 19.0F));
                            const int thickness = cloud_scale * (2 + static_cast<int>(random_unit(
                                 candidate_x,
                                 candidate_z,
                                 settings.seed ^ 0x319CB47DU ^ cluster_salt) * 6.0F));
                            const int style = static_cast<int>(random_unit(
                                candidate_x,
                                candidate_z,
                                settings.seed ^ 0x94D049BBU ^ cluster_salt) * 16.0F) % 16;
                            const auto inside_rectangle = [along, across](
                                                              const int minimum_along,
                                                              const int maximum_along,
                                                              const int minimum_across,
                                                              const int maximum_across) {
                                return along >= minimum_along && along < maximum_along &&
                                       across >= minimum_across && across < maximum_across;
                            };
                            const bool main_rectangle =
                                inside_rectangle(0, length, 0, thickness);
                            const int branch_direction = (style & 1) == 0 ? -1 : 1;
                            const int opposite_direction = -branch_direction;
                            const bool leading_lobe = (style & 2) != 0 && inside_rectangle(
                                cloud_scale,
                                std::min(length, (5 + style % 7) * cloud_scale),
                                branch_direction < 0 ? -(2 + style % 4) * cloud_scale : 0,
                                branch_direction < 0
                                    ? thickness
                                    : thickness + (2 + style % 4) * cloud_scale);
                            const bool middle_lobe = (style & 4) != 0 && inside_rectangle(
                                length / 3,
                                std::min(
                                    length + cloud_scale,
                                    length / 3 + (4 + style % 6) * cloud_scale),
                                opposite_direction < 0
                                    ? -(2 + (style / 2) % 4) * cloud_scale
                                    : 0,
                                opposite_direction < 0
                                    ? thickness
                                    : thickness + (2 + (style / 2) % 4) * cloud_scale);
                            const bool trailing_lobe = (style & 8) != 0 && inside_rectangle(
                                std::max(0, length - (6 + style % 3) * cloud_scale),
                                length + (1 + style % 3) * cloud_scale,
                                -(1 + style % 3) * cloud_scale,
                                thickness + (1 + (style + 1) % 4) * cloud_scale);
                            inside_cloud = main_rectangle || leading_lobe || middle_lobe ||
                                           trailing_lobe;
                        }
                    }
                }
                if (inside_cloud) {
                    for (int layer = 0; layer < 2; ++layer) {
                        const int cloud_y = local_cloud_y + layer;
                        if (cloud_y < dimensions.height &&
                            world.block_at(x, cloud_y, z) == BlockType::air) {
                            world.set_block_untracked(x, cloud_y, z, BlockType::cloud);
                        }
                    }
                }
            }
        };
        if (use_worker_pool) {
            core::parallel_for(static_cast<std::size_t>(dimensions.depth), generate_cloud_row);
        } else {
            for (int row = 0; row < dimensions.depth; ++row) {
                generate_cloud_row(static_cast<std::size_t>(row));
            }
        }
    }

    if (build_section_index) {
        world.rebuild_section_occupancy();
    }
    return world;
}

float WorldGenerator::fractal_noise(const float x, const float z, const std::uint32_t seed) noexcept {
    float result = 0.0F;
    float amplitude = 1.0F;
    float frequency = 1.0F;
    float amplitude_sum = 0.0F;

    for (std::uint32_t octave = 0; octave < 5; ++octave) {
        result += value_noise(x * frequency, z * frequency, seed + octave * 0x9E3779B9U) * amplitude;
        amplitude_sum += amplitude;
        amplitude *= 0.5F;
        frequency *= 2.0F;
    }
    return result / amplitude_sum;
}

float WorldGenerator::random_unit(const int x, const int z, const std::uint32_t seed) noexcept {
    return random_unit_value(x, z, seed);
}

void WorldGenerator::place_tree(
    World& world,
    const int x,
    const int ground_y,
    const int z,
    const int trunk_height) {
    const int crown_y = ground_y + trunk_height;
    if (!world.contains(x - 2, ground_y + 1, z - 2) ||
        !world.contains(x + 2, crown_y + 2, z + 2)) {
        return;
    }

    for (int y = ground_y + 1; y <= crown_y; ++y) {
        world.set_block_untracked(x, y, z, BlockType::log);
    }

    for (int y = crown_y - 2; y <= crown_y + 2; ++y) {
        const int layer_distance = std::abs(y - crown_y);
        const int radius = layer_distance == 2 ? 1 : 2;
        for (int offset_z = -radius; offset_z <= radius; ++offset_z) {
            for (int offset_x = -radius; offset_x <= radius; ++offset_x) {
                const bool corner = std::abs(offset_x) == radius && std::abs(offset_z) == radius;
                if ((corner && layer_distance > 0) ||
                    world.block_at(x + offset_x, y, z + offset_z) != BlockType::air) {
                    continue;
                }
                world.set_block_untracked(
                    x + offset_x, y, z + offset_z, BlockType::leaves);
            }
        }
    }
}

}  // namespace vulkancraft::world
