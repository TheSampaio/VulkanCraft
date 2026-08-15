#include "vulkancraft/world/World.hpp"

#include <algorithm>
#include <stdexcept>

#include "vulkancraft/core/Parallel.hpp"

namespace vulkancraft::world {

World::World(const WorldDimensions dimensions) : dimensions_(dimensions) {
    if (dimensions.width <= 0 || dimensions.height <= 0 || dimensions.depth <= 0) {
        throw std::invalid_argument("World dimensions must be positive");
    }

    const auto width = static_cast<std::size_t>(dimensions.width);
    const auto height = static_cast<std::size_t>(dimensions.height);
    const auto depth = static_cast<std::size_t>(dimensions.depth);
    blocks_.resize(width * height * depth, BlockType::air);
    water_levels_.resize(width * height * depth, no_water_level);
    section_block_counts_ = std::vector<std::uint32_t>(
        static_cast<std::size_t>(section_count_x()) *
        static_cast<std::size_t>(section_count_y()) *
        static_cast<std::size_t>(section_count_z()), 0U);
}

const WorldDimensions& World::dimensions() const noexcept {
    return dimensions_;
}

bool World::contains(const int x, const int y, const int z) const noexcept {
    return x >= 0 && x < dimensions_.width && y >= 0 && y < dimensions_.height && z >= 0 &&
           z < dimensions_.depth;
}

BlockType World::block_at(const int x, const int y, const int z) const noexcept {
    return contains(x, y, z) ? blocks_[index_of(x, y, z)] : BlockType::air;
}

std::uint8_t World::water_level_at(const int x, const int y, const int z) const noexcept {
    return contains(x, y, z) && block_at(x, y, z) == BlockType::water
               ? water_levels_[index_of(x, y, z)]
               : no_water_level;
}

void World::set_block(const int x, const int y, const int z, const BlockType block) {
    if (!contains(x, y, z)) {
        throw std::out_of_range("Block coordinates are outside the world");
    }
    BlockType& current = blocks_[index_of(x, y, z)];
    const bool was_air = current == BlockType::air;
    const bool becomes_air = block == BlockType::air;
    if (was_air != becomes_air) {
        auto& count = section_block_counts_[section_index_of(
            x / section_size, y / section_size, z / section_size)];
        if (becomes_air) {
            --count;
        } else {
            ++count;
        }
    }
    current = block;
    water_levels_[index_of(x, y, z)] =
        block == BlockType::water ? water_source_level : no_water_level;
}

void World::set_water(
    const int x,
    const int y,
    const int z,
    const std::uint8_t level) {
    if (level > falling_water_level) {
        throw std::invalid_argument("Water level must be in the inclusive range zero through eight");
    }
    set_block(x, y, z, BlockType::water);
    water_levels_[index_of(x, y, z)] = level;
}

void World::set_block_untracked(
    const int x,
    const int y,
    const int z,
    const BlockType block) {
    if (!contains(x, y, z)) {
        throw std::out_of_range("Block coordinates are outside the world");
    }
    blocks_[index_of(x, y, z)] = block;
    water_levels_[index_of(x, y, z)] =
        block == BlockType::water ? water_source_level : no_water_level;
}

void World::rebuild_section_occupancy() {
    core::parallel_for(section_block_counts_.size(), [this](const std::size_t section_index) {
        const int sections_per_layer = section_count_x() * section_count_z();
        const int section_y = static_cast<int>(section_index) / sections_per_layer;
        const int layer_index = static_cast<int>(section_index) % sections_per_layer;
        const int section_z = layer_index / section_count_x();
        const int section_x = layer_index % section_count_x();
        const int minimum_x = section_x * section_size;
        const int minimum_y = section_y * section_size;
        const int minimum_z = section_z * section_size;
        const int maximum_x = std::min(minimum_x + section_size, dimensions_.width);
        const int maximum_y = std::min(minimum_y + section_size, dimensions_.height);
        const int maximum_z = std::min(minimum_z + section_size, dimensions_.depth);
        std::uint32_t count = 0U;
        for (int y = minimum_y; y < maximum_y; ++y) {
            for (int z = minimum_z; z < maximum_z; ++z) {
                for (int x = minimum_x; x < maximum_x; ++x) {
                    count += block_at(x, y, z) == BlockType::air ? 0U : 1U;
                }
            }
        }
        section_block_counts_[section_index] = count;
    });
}

void World::copy_columns_from(
    const World& source,
    const int source_x,
    const int source_z,
    const int destination_x,
    const int destination_z,
    const int width,
    const int depth) {
    if (width == 0 || depth == 0) {
        return;
    }
    if (source.dimensions_.height != dimensions_.height || width < 0 || depth < 0 ||
        !source.contains(source_x, 0, source_z) ||
        !source.contains(source_x + width - 1, dimensions_.height - 1, source_z + depth - 1) ||
        !contains(destination_x, 0, destination_z) ||
        !contains(
            destination_x + width - 1,
            dimensions_.height - 1,
            destination_z + depth - 1)) {
        throw std::out_of_range("Copied world columns are outside the source or destination");
    }
    for (int y = 0; y < dimensions_.height; ++y) {
        for (int z = 0; z < depth; ++z) {
            const auto source_begin = source.blocks_.begin() + static_cast<std::ptrdiff_t>(
                source.index_of(source_x, y, source_z + z));
            auto destination_begin = blocks_.begin() + static_cast<std::ptrdiff_t>(
                index_of(destination_x, y, destination_z + z));
            std::copy_n(source_begin, width, destination_begin);
            const auto source_water_begin = source.water_levels_.begin() +
                                            static_cast<std::ptrdiff_t>(
                                                source.index_of(source_x, y, source_z + z));
            auto destination_water_begin = water_levels_.begin() + static_cast<std::ptrdiff_t>(
                index_of(destination_x, y, destination_z + z));
            std::copy_n(source_water_begin, width, destination_water_begin);
        }
    }
}

std::size_t World::size() const noexcept {
    return blocks_.size();
}

bool World::section_contains_blocks(
    const int section_x,
    const int section_y,
    const int section_z) const noexcept {
    if (section_x < 0 || section_x >= section_count_x() || section_y < 0 ||
        section_y >= section_count_y() || section_z < 0 || section_z >= section_count_z()) {
        return false;
    }
    return section_block_counts_[section_index_of(section_x, section_y, section_z)] != 0U;
}

int World::section_count_x() const noexcept {
    return (dimensions_.width + section_size - 1) / section_size;
}

int World::section_count_y() const noexcept {
    return (dimensions_.height + section_size - 1) / section_size;
}

int World::section_count_z() const noexcept {
    return (dimensions_.depth + section_size - 1) / section_size;
}

std::size_t World::index_of(const int x, const int y, const int z) const noexcept {
    const auto width = static_cast<std::size_t>(dimensions_.width);
    const auto depth = static_cast<std::size_t>(dimensions_.depth);
    return static_cast<std::size_t>(y) * width * depth + static_cast<std::size_t>(z) * width +
           static_cast<std::size_t>(x);
}

std::size_t World::section_index_of(
    const int section_x,
    const int section_y,
    const int section_z) const noexcept {
    return static_cast<std::size_t>(section_y) *
               static_cast<std::size_t>(section_count_x() * section_count_z()) +
           static_cast<std::size_t>(section_z * section_count_x() + section_x);
}

}  // namespace vulkancraft::world
