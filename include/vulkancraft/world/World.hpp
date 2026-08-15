#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vulkancraft/world/Block.hpp"

namespace vulkancraft::world {

/** Defines the voxel dimensions of a world. */
struct WorldDimensions {
    int width{96};
    int height{48};
    int depth{96};
};

/** Owns a dense, bounds-safe three-dimensional voxel field. */
class World final {
public:
    /** Edge length of one occupancy section used to skip empty voxel ranges. */
    static constexpr int section_size = 16;

    /**
     * Creates an air-filled world.
     *
     * @param dimensions Positive voxel dimensions.
     * @throws std::invalid_argument If any dimension is not positive.
     */
    explicit World(WorldDimensions dimensions);

    World(const World&) = delete;
    World& operator=(const World&) = delete;
    World(World&&) noexcept = default;
    World& operator=(World&&) noexcept = default;

    /**
     * Returns the configured dimensions.
     *
     * @return Immutable world dimensions.
     */
    [[nodiscard]] const WorldDimensions& dimensions() const noexcept;

    /**
     * Tests whether voxel coordinates are inside the world.
     *
     * @param x Horizontal x coordinate.
     * @param y Vertical coordinate.
     * @param z Horizontal z coordinate.
     * @return True when all three coordinates are valid.
     */
    [[nodiscard]] bool contains(int x, int y, int z) const noexcept;

    /**
     * Reads one block, treating positions outside the world as air.
     *
     * @param x Horizontal x coordinate.
     * @param y Vertical coordinate.
     * @param z Horizontal z coordinate.
     * @return Stored block or air when outside the world.
     */
    [[nodiscard]] BlockType block_at(int x, int y, int z) const noexcept;

    /**
     * Reads the water depth metadata stored at one voxel.
     *
     * @param x Horizontal x coordinate.
     * @param y Vertical coordinate.
     * @param z Horizontal z coordinate.
     * @return Zero through seven for horizontal water, eight for falling water,
     *     or `no_water_level` for non-water blocks and positions outside the world.
     */
    [[nodiscard]] std::uint8_t water_level_at(int x, int y, int z) const noexcept;

    /**
     * Replaces one block.
     *
     * @param x Horizontal x coordinate.
     * @param y Vertical coordinate.
     * @param z Horizontal z coordinate.
     * @param block New material.
     * @throws std::out_of_range If the coordinates are outside the world.
     */
    void set_block(int x, int y, int z, BlockType block);

    /**
     * Stores a water voxel with an explicit source, flow, or falling level.
     *
     * @param x Horizontal x coordinate.
     * @param y Vertical coordinate.
     * @param z Horizontal z coordinate.
     * @param level Zero through seven for horizontal water or eight for falling water.
     */
    void set_water(int x, int y, int z, std::uint8_t level);

    /**
     * Replaces a block without updating section metadata during parallel bulk generation.
     *
     * Call `rebuild_section_occupancy` after the complete bulk write. Coordinates
     * written concurrently must still be distinct.
     *
     * @param x Horizontal x coordinate.
     * @param y Vertical coordinate.
     * @param z Horizontal z coordinate.
     * @param block New material.
     */
    void set_block_untracked(int x, int y, int z, BlockType block);

    /** Recalculates all section occupancy metadata after bulk voxel writes. */
    void rebuild_section_occupancy();

    /**
     * Copies a horizontal rectangle for every y level without rebuilding metadata.
     *
     * @param source Immutable source world with the same vertical dimension.
     * @param source_x First source x coordinate.
     * @param source_z First source z coordinate.
     * @param destination_x First destination x coordinate.
     * @param destination_z First destination z coordinate.
     * @param width Number of columns copied along x.
     * @param depth Number of columns copied along z.
     */
    void copy_columns_from(
        const World& source,
        int source_x,
        int source_z,
        int destination_x,
        int destination_z,
        int width,
        int depth);

    /**
     * Returns the number of allocated voxels.
     *
     * @return Total voxel count.
     */
    [[nodiscard]] std::size_t size() const noexcept;

    /**
     * Reports whether one 16-cubed storage section contains a non-air block.
     *
     * @param section_x Section coordinate along x.
     * @param section_y Section coordinate along y.
     * @param section_z Section coordinate along z.
     * @return True when the section contains at least one non-air voxel.
     */
    [[nodiscard]] bool section_contains_blocks(
        int section_x,
        int section_y,
        int section_z) const noexcept;

    /** Returns the number of occupancy sections along x. */
    [[nodiscard]] int section_count_x() const noexcept;

    /** Returns the number of occupancy sections along y. */
    [[nodiscard]] int section_count_y() const noexcept;

    /** Returns the number of occupancy sections along z. */
    [[nodiscard]] int section_count_z() const noexcept;

private:
    /** Converts valid coordinates to the contiguous storage index. */
    [[nodiscard]] std::size_t index_of(int x, int y, int z) const noexcept;

    /** Converts valid section coordinates to the occupancy index. */
    [[nodiscard]] std::size_t section_index_of(
        int section_x,
        int section_y,
        int section_z) const noexcept;

    WorldDimensions dimensions_;
    std::vector<BlockType> blocks_;
    std::vector<std::uint8_t> water_levels_;
    std::vector<std::uint32_t> section_block_counts_;
};

}  // namespace vulkancraft::world
