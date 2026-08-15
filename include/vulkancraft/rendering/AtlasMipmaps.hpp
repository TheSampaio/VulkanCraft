#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vulkancraft::rendering {

/** Describes one tightly packed RGBA atlas mip level. */
struct AtlasMipLevel {
    std::size_t byte_offset{};
    std::uint32_t width{};
    std::uint32_t height{};
};

/** Owns a sprite-safe RGBA atlas mip chain and its upload layout. */
struct AtlasMipChain {
    std::vector<std::uint8_t> pixels;
    std::vector<AtlasMipLevel> levels;
};

/**
 * Builds a complete mip chain without filtering across sprite boundaries.
 *
 * Every mip reduces each tile independently, so adjacent sprites can never
 * bleed into one another. RGB channels use alpha-weighted averaging to retain
 * clean colors around cutout foliage.
 *
 * @param base_pixels Tightly packed RGBA8 base-level pixels.
 * @param width Base-level atlas width in pixels.
 * @param height Base-level atlas height in pixels.
 * @param tile_size Square sprite size in pixels; it must be a power of two.
 * @return Packed mip pixels and one upload description per level.
 * @throws std::invalid_argument If the input or atlas grid is invalid.
 */
[[nodiscard]] AtlasMipChain build_atlas_mip_chain(
    std::span<const std::uint8_t> base_pixels,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t tile_size);

}  // namespace vulkancraft::rendering
