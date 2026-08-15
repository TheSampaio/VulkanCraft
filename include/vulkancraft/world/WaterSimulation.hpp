#pragma once

#include <cstddef>
#include <deque>
#include <unordered_set>
#include <vector>

#include "vulkancraft/world/World.hpp"

namespace vulkancraft::world {

/** Integer voxel coordinate used by scheduled world simulations. */
struct VoxelPosition {
    int x{};
    int y{};
    int z{};

    bool operator==(const VoxelPosition&) const = default;
};

/** Hashes a voxel coordinate for duplicate-free scheduled updates. */
struct VoxelPositionHash {
    /**
     * Produces a stable hash for one voxel coordinate.
     *
     * @param position Coordinate to hash.
     * @return Hash value suitable for unordered containers.
     */
    [[nodiscard]] std::size_t operator()(const VoxelPosition& position) const noexcept;
};

/** Advances Minecraft-like water propagation in fixed scheduled ticks. */
class WaterSimulation final {
public:
    /** Java Edition water update interval: five 20 Hz game ticks. */
    static constexpr float tick_interval_seconds = 0.25F;

    /** Maximum horizontal distance beyond a source block. */
    static constexpr std::uint8_t maximum_horizontal_distance = 7U;

    /** Maximum horizontal search distance used to prefer nearby drops. */
    static constexpr int downward_search_distance = 4;

    /** Clears scheduled work when the active streamed voxel window changes. */
    void reset() noexcept;

    /**
     * Schedules water affected by a changed voxel.
     *
     * @param world Current voxel field.
     * @param position Changed local voxel coordinate.
     */
    void notify_block_changed(const World& world, VoxelPosition position);

    /**
     * Advances scheduled water updates using fixed 0.25-second ticks.
     *
     * Newly created water is deferred until the next tick, making falling and
     * horizontal propagation advance one block at a time.
     *
     * @param world Mutable voxel field.
     * @param delta_seconds Elapsed real time since the previous frame.
     * @return Local voxel positions changed by fluid propagation.
     */
    [[nodiscard]] std::vector<VoxelPosition> advance(World& world, float delta_seconds);

private:
    /** Adds one valid coordinate to the duplicate-free pending queue. */
    void schedule(const World& world, VoxelPosition position);

    /** Finds the shortest horizontal route to a downward opening. */
    [[nodiscard]] static int distance_to_drop(
        const World& world,
        VoxelPosition start) noexcept;

    std::deque<VoxelPosition> pending_;
    std::unordered_set<VoxelPosition, VoxelPositionHash> scheduled_;
    float accumulated_seconds_{};
};

}  // namespace vulkancraft::world
