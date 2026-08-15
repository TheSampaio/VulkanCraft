#pragma once

#include <cstdint>

#include "vulkancraft/world/World.hpp"

namespace vulkancraft::world {

/** Controls deterministic procedural terrain generation. */
struct WorldGenerationSettings {
    std::uint32_t seed{0xC0FFEEU};
    int minimum_y{-64};
    int sea_level{62};
    int dirt_depth{4};
    float tree_probability{0.012F};
    int cloud_level{136};
};

/** Builds voxel terrain without depending on graphics or global state. */
class WorldGenerator final {
public:
    /**
     * Generates terrain and vegetation into a new world.
     *
     * @param dimensions Target voxel dimensions.
     * @param settings Reproducible generation parameters.
     * @return Fully generated world.
     */
    [[nodiscard]] World generate(
        WorldDimensions dimensions,
        const WorldGenerationSettings& settings = {}) const;

    /**
     * Generates a region using global horizontal coordinates.
     *
     * Adjacent regions are deterministic and agree at their shared boundary.
     *
     * @param dimensions Target local voxel dimensions.
     * @param origin_x Global x coordinate represented by local x zero.
     * @param origin_z Global z coordinate represented by local z zero.
     * @param settings Reproducible generation parameters.
     * @param build_section_index Whether to build occupancy metadata before returning.
     * @param use_worker_pool Whether row generation may use the shared four-worker pool.
     * @param surface_only Whether to omit underground voxels hidden by surface LOD.
     * @return Fully generated local world region.
     */
    [[nodiscard]] World generate_region(
        WorldDimensions dimensions,
        int origin_x,
        int origin_z,
        const WorldGenerationSettings& settings = {},
        bool build_section_index = true,
        bool use_worker_pool = true,
        bool surface_only = false) const;

private:
    /** Samples multi-octave value noise at world coordinates. */
    [[nodiscard]] static float fractal_noise(float x, float z, std::uint32_t seed) noexcept;

    /** Returns a deterministic pseudo-random scalar in the inclusive unit range. */
    [[nodiscard]] static float random_unit(int x, int z, std::uint32_t seed) noexcept;

    /** Places one tree when the complete canopy fits inside the world. */
    static void place_tree(World& world, int x, int local_ground_y, int z, int trunk_height);
};

}  // namespace vulkancraft::world
