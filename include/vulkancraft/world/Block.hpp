#pragma once

#include <cstdint>

namespace vulkancraft::world {

/** Identifies the material stored in one voxel. */
enum class BlockType : std::uint8_t {
    air = 0,
    grass,
    dirt,
    sand,
    stone,
    log,
    leaves,
    water,
    cloud,
};

/** Fluid depth stored beside water voxels, matching Java Edition water levels. */
inline constexpr std::uint8_t water_source_level = 0U;
inline constexpr std::uint8_t maximum_water_flow_level = 7U;
inline constexpr std::uint8_t falling_water_level = 8U;
inline constexpr std::uint8_t no_water_level = 0xFFU;

/**
 * Reports whether a block occupies space and should produce geometry.
 *
 * @param block Block material to inspect.
 * @return True for terrain and vegetation that collide with the player.
 */
[[nodiscard]] constexpr bool is_solid(const BlockType block) noexcept {
    return block != BlockType::air && block != BlockType::water && block != BlockType::cloud;
}

/** Reports whether a block belongs in the blended rendering pass. */
[[nodiscard]] constexpr bool is_transparent(const BlockType block) noexcept {
    return block == BlockType::water || block == BlockType::cloud;
}

/**
 * Reports whether a block has visible geometry.
 *
 * @param block Block material to inspect.
 * @return True for terrain, vegetation, and water.
 */
[[nodiscard]] constexpr bool is_renderable(const BlockType block) noexcept {
    return block != BlockType::air;
}

}  // namespace vulkancraft::world
