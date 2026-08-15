#include "vulkancraft/world/WaterSimulation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace vulkancraft::world {
namespace {

constexpr std::array<VoxelPosition, 4> horizontal_directions{{
    {1, 0, 0},
    {-1, 0, 0},
    {0, 0, 1},
    {0, 0, -1},
}};

[[nodiscard]] VoxelPosition offset(
    const VoxelPosition position,
    const VoxelPosition direction) noexcept {
    return {
        position.x + direction.x,
        position.y + direction.y,
        position.z + direction.z,
    };
}

}  // namespace

std::size_t VoxelPositionHash::operator()(const VoxelPosition& position) const noexcept {
    const auto x = static_cast<std::uint32_t>(position.x);
    const auto y = static_cast<std::uint32_t>(position.y);
    const auto z = static_cast<std::uint32_t>(position.z);
    std::uint32_t hash = x * 0x9E3779B9U;
    hash ^= y + 0x85EBCA6BU + (hash << 6U) + (hash >> 2U);
    hash ^= z + 0xC2B2AE35U + (hash << 6U) + (hash >> 2U);
    return static_cast<std::size_t>(hash);
}

void WaterSimulation::reset() noexcept {
    pending_.clear();
    scheduled_.clear();
    accumulated_seconds_ = 0.0F;
}

void WaterSimulation::notify_block_changed(
    const World& world,
    const VoxelPosition position) {
    schedule(world, position);
    schedule(world, {position.x, position.y + 1, position.z});
    for (const VoxelPosition direction : horizontal_directions) {
        schedule(world, offset(position, direction));
    }
}

std::vector<VoxelPosition> WaterSimulation::advance(
    World& world,
    const float delta_seconds) {
    if (pending_.empty()) {
        accumulated_seconds_ = 0.0F;
        return {};
    }
    accumulated_seconds_ += std::max(delta_seconds, 0.0F);
    std::vector<VoxelPosition> changed;
    while (accumulated_seconds_ >= tick_interval_seconds && !pending_.empty()) {
        accumulated_seconds_ -= tick_interval_seconds;
        const std::size_t scheduled_this_tick = pending_.size();
        for (std::size_t update = 0; update < scheduled_this_tick; ++update) {
            const VoxelPosition position = pending_.front();
            pending_.pop_front();
            scheduled_.erase(position);
            if (world.block_at(position.x, position.y, position.z) != BlockType::water) {
                continue;
            }

            const std::uint8_t current_level =
                world.water_level_at(position.x, position.y, position.z);
            if (current_level == falling_water_level) {
                if (world.block_at(position.x, position.y + 1, position.z) != BlockType::water) {
                    world.set_block(position.x, position.y, position.z, BlockType::air);
                    changed.push_back(position);
                    notify_block_changed(world, position);
                    continue;
                }
            } else if (current_level > water_source_level) {
                bool supported = false;
                for (const VoxelPosition direction : horizontal_directions) {
                    const VoxelPosition neighbor = offset(position, direction);
                    const std::uint8_t neighbor_level =
                        world.water_level_at(neighbor.x, neighbor.y, neighbor.z);
                    supported = supported || neighbor_level < current_level ||
                                (current_level == 1U && neighbor_level == falling_water_level);
                }
                if (!supported) {
                    world.set_block(position.x, position.y, position.z, BlockType::air);
                    changed.push_back(position);
                    notify_block_changed(world, position);
                    continue;
                }
            }

            const VoxelPosition below{position.x, position.y - 1, position.z};
            if (world.contains(below.x, below.y, below.z) &&
                world.block_at(below.x, below.y, below.z) == BlockType::air) {
                world.set_water(below.x, below.y, below.z, falling_water_level);
                changed.push_back(below);
                schedule(world, below);
                schedule(world, position);
                continue;
            }

            const std::uint8_t next_level = current_level == falling_water_level
                                                ? 1U
                                                : static_cast<std::uint8_t>(current_level + 1U);
            if (next_level > maximum_horizontal_distance) {
                continue;
            }

            std::array<int, horizontal_directions.size()> weights{};
            weights.fill(std::numeric_limits<int>::max());
            int best_weight = std::numeric_limits<int>::max();
            for (std::size_t index = 0; index < horizontal_directions.size(); ++index) {
                const VoxelPosition candidate = offset(position, horizontal_directions[index]);
                if (!world.contains(candidate.x, candidate.y, candidate.z) ||
                    world.block_at(candidate.x, candidate.y, candidate.z) != BlockType::air) {
                    continue;
                }
                weights[index] = distance_to_drop(world, candidate);
                best_weight = std::min(best_weight, weights[index]);
            }

            const bool found_drop = best_weight <= downward_search_distance;
            for (std::size_t index = 0; index < horizontal_directions.size(); ++index) {
                if (weights[index] == std::numeric_limits<int>::max() ||
                    (found_drop && weights[index] != best_weight)) {
                    continue;
                }
                const VoxelPosition candidate = offset(position, horizontal_directions[index]);
                world.set_water(candidate.x, candidate.y, candidate.z, next_level);
                changed.push_back(candidate);
                schedule(world, candidate);
            }
        }
    }
    if (pending_.empty()) {
        accumulated_seconds_ = 0.0F;
    }
    return changed;
}

void WaterSimulation::schedule(const World& world, const VoxelPosition position) {
    if (!world.contains(position.x, position.y, position.z) ||
        world.block_at(position.x, position.y, position.z) != BlockType::water ||
        scheduled_.contains(position)) {
        return;
    }
    scheduled_.insert(position);
    pending_.push_back(position);
}

int WaterSimulation::distance_to_drop(
    const World& world,
    const VoxelPosition start) noexcept {
    struct SearchNode {
        VoxelPosition position;
        int distance{};
    };
    std::array<SearchNode, 64> queue{};
    std::array<VoxelPosition, 64> visited{};
    std::size_t head = 0U;
    std::size_t tail = 1U;
    std::size_t visited_count = 1U;
    queue[0] = {start, 0};
    visited[0] = start;
    while (head < tail) {
        const SearchNode node = queue[head++];
        if (world.contains(node.position.x, node.position.y - 1, node.position.z) &&
            world.block_at(node.position.x, node.position.y - 1, node.position.z) ==
                BlockType::air) {
            return node.distance;
        }
        if (node.distance >= downward_search_distance) {
            continue;
        }
        for (const VoxelPosition direction : horizontal_directions) {
            const VoxelPosition candidate = offset(node.position, direction);
            if (!world.contains(candidate.x, candidate.y, candidate.z) ||
                world.block_at(candidate.x, candidate.y, candidate.z) != BlockType::air) {
                continue;
            }
            const bool already_visited = std::find(
                                             visited.begin(),
                                             visited.begin() +
                                                 static_cast<std::ptrdiff_t>(visited_count),
                                             candidate) !=
                                         visited.begin() +
                                             static_cast<std::ptrdiff_t>(visited_count);
            if (already_visited || tail >= queue.size()) {
                continue;
            }
            visited[visited_count++] = candidate;
            queue[tail++] = {candidate, node.distance + 1};
        }
    }
    return downward_search_distance + 1;
}

}  // namespace vulkancraft::world
