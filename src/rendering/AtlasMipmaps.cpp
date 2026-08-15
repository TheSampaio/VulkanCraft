#include "vulkancraft/rendering/AtlasMipmaps.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace vulkancraft::rendering {
namespace {

constexpr std::size_t channel_count = 4U;

[[nodiscard]] std::size_t byte_count(
    const std::uint32_t width,
    const std::uint32_t height) {
    const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixels > std::numeric_limits<std::size_t>::max() / channel_count) {
        throw std::invalid_argument("Atlas dimensions overflow addressable memory");
    }
    return pixels * channel_count;
}

}  // namespace

AtlasMipChain build_atlas_mip_chain(
    const std::span<const std::uint8_t> base_pixels,
    const std::uint32_t width,
    const std::uint32_t height,
    const std::uint32_t tile_size) {
    if (width == 0U || height == 0U || tile_size == 0U ||
        (tile_size & (tile_size - 1U)) != 0U || width % tile_size != 0U ||
        height % tile_size != 0U || base_pixels.size() != byte_count(width, height)) {
        throw std::invalid_argument("Invalid sprite atlas mip input");
    }

    AtlasMipChain result;
    std::uint32_t mip_count = 1U;
    for (std::uint32_t size = tile_size; size > 1U; size /= 2U) {
        ++mip_count;
    }
    result.levels.reserve(mip_count);

    std::size_t total_bytes = 0U;
    for (std::uint32_t level = 0U; level < mip_count; ++level) {
        total_bytes += byte_count(width >> level, height >> level);
    }
    result.pixels.reserve(total_bytes);
    result.levels.push_back({0U, width, height});
    result.pixels.insert(result.pixels.end(), base_pixels.begin(), base_pixels.end());

    const std::uint32_t tile_columns = width / tile_size;
    const std::uint32_t tile_rows = height / tile_size;
    for (std::uint32_t level = 1U; level < mip_count; ++level) {
        const AtlasMipLevel& previous = result.levels.back();
        const std::uint32_t level_width = width >> level;
        const std::uint32_t level_height = height >> level;
        const std::uint32_t level_tile_size = tile_size >> level;
        const std::size_t level_offset = result.pixels.size();
        result.levels.push_back({level_offset, level_width, level_height});
        result.pixels.resize(level_offset + byte_count(level_width, level_height));

        for (std::uint32_t tile_y = 0U; tile_y < tile_rows; ++tile_y) {
            for (std::uint32_t tile_x = 0U; tile_x < tile_columns; ++tile_x) {
                for (std::uint32_t y = 0U; y < level_tile_size; ++y) {
                    for (std::uint32_t x = 0U; x < level_tile_size; ++x) {
                        const std::uint32_t destination_x = tile_x * level_tile_size + x;
                        const std::uint32_t destination_y = tile_y * level_tile_size + y;
                        const std::size_t destination = level_offset +
                            (static_cast<std::size_t>(destination_y) * level_width + destination_x) *
                                channel_count;

                        std::uint32_t alpha_sum = 0U;
                        std::uint32_t red_sum = 0U;
                        std::uint32_t green_sum = 0U;
                        std::uint32_t blue_sum = 0U;
                        for (std::uint32_t offset_y = 0U; offset_y < 2U; ++offset_y) {
                            for (std::uint32_t offset_x = 0U; offset_x < 2U; ++offset_x) {
                                const std::uint32_t source_x = tile_x * level_tile_size * 2U +
                                                               x * 2U + offset_x;
                                const std::uint32_t source_y = tile_y * level_tile_size * 2U +
                                                               y * 2U + offset_y;
                                const std::size_t source = previous.byte_offset +
                                    (static_cast<std::size_t>(source_y) * previous.width + source_x) *
                                        channel_count;
                                const std::uint32_t alpha = result.pixels[source + 3U];
                                alpha_sum += alpha;
                                red_sum += static_cast<std::uint32_t>(result.pixels[source]) * alpha;
                                green_sum +=
                                    static_cast<std::uint32_t>(result.pixels[source + 1U]) * alpha;
                                blue_sum +=
                                    static_cast<std::uint32_t>(result.pixels[source + 2U]) * alpha;
                            }
                        }

                        if (alpha_sum > 0U) {
                            result.pixels[destination] = static_cast<std::uint8_t>(red_sum / alpha_sum);
                            result.pixels[destination + 1U] =
                                static_cast<std::uint8_t>(green_sum / alpha_sum);
                            result.pixels[destination + 2U] =
                                static_cast<std::uint8_t>(blue_sum / alpha_sum);
                        }
                        result.pixels[destination + 3U] =
                            static_cast<std::uint8_t>((alpha_sum + 2U) / 4U);
                    }
                }
            }
        }
    }

    return result;
}

}  // namespace vulkancraft::rendering
