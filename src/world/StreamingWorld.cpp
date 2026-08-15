#include "vulkancraft/world/StreamingWorld.hpp"

#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "vulkancraft/core/Parallel.hpp"

namespace vulkancraft::world {
namespace {

[[nodiscard]] int minimum_window_offset(const WorldStreamingSettings& settings) noexcept {
    return -(settings.render_distance / 2);
}

[[nodiscard]] ChunkCoordinate window_coordinate(
    const ChunkCoordinate center,
    const WorldStreamingSettings& settings,
    const int x,
    const int z) noexcept {
    const int minimum_offset = minimum_window_offset(settings);
    return {center.x + minimum_offset + x, center.z + minimum_offset + z};
}

[[nodiscard]] bool contains_chunk(
    const ChunkCoordinate center,
    const WorldStreamingSettings& settings,
    const ChunkCoordinate coordinate) noexcept {
    const int minimum_offset = minimum_window_offset(settings);
    const int maximum_offset = minimum_offset + settings.render_distance - 1;
    return coordinate.x >= center.x + minimum_offset &&
           coordinate.x <= center.x + maximum_offset &&
           coordinate.z >= center.z + minimum_offset &&
           coordinate.z <= center.z + maximum_offset;
}

[[nodiscard]] bool uses_surface_lod(
    const ChunkCoordinate center,
    const ChunkCoordinate coordinate,
    const WorldStreamingSettings& settings) noexcept {
    return std::max(
               std::abs(coordinate.x - center.x),
               std::abs(coordinate.z - center.z)) > settings.full_detail_radius;
}

struct LodProfile {
    bool surface{};
    bool minimum_x{};
    bool maximum_x{};
    bool minimum_z{};
    bool maximum_z{};

    bool operator==(const LodProfile&) const = default;
};

[[nodiscard]] LodProfile lod_profile(
    const ChunkCoordinate center,
    const ChunkCoordinate coordinate,
    const WorldStreamingSettings& settings) noexcept {
    const bool surface = uses_surface_lod(center, coordinate, settings);
    return {
        .surface = surface,
        .minimum_x = surface &&
                     !uses_surface_lod(center, {coordinate.x - 1, coordinate.z}, settings),
        .maximum_x = surface &&
                     !uses_surface_lod(center, {coordinate.x + 1, coordinate.z}, settings),
        .minimum_z = surface &&
                     !uses_surface_lod(center, {coordinate.x, coordinate.z - 1}, settings),
        .maximum_z = surface &&
                     !uses_surface_lod(center, {coordinate.x, coordinate.z + 1}, settings),
    };
}

[[nodiscard]] int floor_divide(const int value, const int divisor) noexcept {
    const int quotient = value / divisor;
    return value % divisor < 0 ? quotient - 1 : quotient;
}

}  // namespace

StreamingWorld::StreamingWorld(
    WorldGenerationSettings generation,
    WorldStreamingSettings streaming)
    : generation_settings_(generation), streaming_settings_(streaming) {
    if (streaming.chunk_size != 32 || streaming.render_distance < 3 ||
        streaming.full_detail_radius < 1 ||
        streaming.full_detail_radius * 2 + 1 > streaming.render_distance ||
        streaming.world_height != 384 || streaming.minimum_y != -64 ||
        streaming.generation_padding < 3) {
        throw std::invalid_argument(
            "Chunks must be 32 by 32 blocks with Y limits -64 through 319");
    }
    generation_settings_.minimum_y = streaming_settings_.minimum_y;
}

StreamingWorld::~StreamingWorld() {
    if (pending_build_.valid()) {
        pending_build_.wait();
    }
}

WorldMesh StreamingWorld::initialize(const glm::vec3& camera_position) {
    BuildResult initial = rebuild(chunk_at(camera_position), std::nullopt);
    active_world_ = std::move(initial.world);
    active_region_ = initial.region;
    active_chunk_meshes_ = std::move(initial.chunk_meshes);
    center_chunk_ = initial.center;
    return std::move(initial.mesh);
}

std::optional<StreamedChunkUpdate> StreamingWorld::update(
    const glm::vec3& camera_position) {
    const ChunkCoordinate desired_center = chunk_at(camera_position);
    if (pending_build_.valid() &&
        pending_build_.wait_for(std::chrono::seconds{0}) == std::future_status::ready) {
        BuildResult completed = pending_build_.get();
        if (completed.edit_revision != edit_revision()) {
            retired_world_ = std::move(completed.world);
            retired_chunk_meshes_ = std::move(completed.chunk_meshes);
            retired_update_ = StreamedChunkUpdate{
                .mesh = std::move(completed.mesh),
                .replacement_indices = std::move(completed.replacement_chunk_indices),
            };
            request_stream(desired_center);
            return std::nullopt;
        }
        if (completed.world.has_value()) {
            if (center_chunk_.has_value() &&
                completed.chunk_meshes.size() == active_chunk_meshes_.size()) {
                const int count = streaming_settings_.render_distance;
                for (int chunk_z = 0; chunk_z < count; ++chunk_z) {
                    for (int chunk_x = 0; chunk_x < count; ++chunk_x) {
                        const std::size_t new_index = static_cast<std::size_t>(
                            chunk_z * count + chunk_x);
                        if (!completed.chunk_meshes[new_index].vertices.empty()) {
                            continue;
                        }
                        const ChunkCoordinate coordinate = window_coordinate(
                            completed.center, streaming_settings_, chunk_x, chunk_z);
                        const int old_x = coordinate.x -
                                          (center_chunk_->x +
                                           minimum_window_offset(streaming_settings_));
                        const int old_z = coordinate.z -
                                          (center_chunk_->z +
                                           minimum_window_offset(streaming_settings_));
                        if (old_x < 0 || old_x >= count || old_z < 0 || old_z >= count) {
                            throw std::logic_error(
                                "A retained streamed chunk was not present in the active window");
                        }
                        completed.chunk_meshes[new_index] = std::move(
                            active_chunk_meshes_[static_cast<std::size_t>(old_z * count + old_x)]);
                    }
                }
            }
            retired_world_ = std::move(active_world_);
            active_world_ = std::move(completed.world);
            active_region_ = completed.region;
            retired_chunk_meshes_ = std::move(active_chunk_meshes_);
            active_chunk_meshes_ = std::move(completed.chunk_meshes);
            dirty_chunk_indices_.clear();
            center_chunk_ = completed.center;
            water_simulation_.reset();
            const std::scoped_lock edits_lock{edits_mutex_};
            for (const auto& [position, edit] : edits_) {
                if (edit.block != BlockType::water) {
                    continue;
                }
                water_simulation_.notify_block_changed(
                    *active_world_,
                    {position.x - active_region_.world_origin_x,
                     position.y - active_region_.world_origin_y,
                     position.z - active_region_.world_origin_z});
            }
        }
        return StreamedChunkUpdate{
            .mesh = std::move(completed.mesh),
            .replacement_indices = std::move(completed.replacement_chunk_indices),
        };
    }

    if (!pending_build_.valid() &&
        (!center_chunk_.has_value() || *center_chunk_ != desired_center)) {
        request_stream(desired_center);
    }
    return std::nullopt;
}

void StreamingWorld::retire_update(StreamedChunkUpdate&& update) noexcept {
    retired_update_ = std::move(update);
}

BlockType StreamingWorld::block_at(
    const int global_x,
    const int global_y,
    const int global_z) const noexcept {
    if (!active_world_.has_value()) {
        return BlockType::air;
    }
    return active_world_->block_at(
        global_x - active_region_.world_origin_x,
        global_y - active_region_.world_origin_y,
        global_z - active_region_.world_origin_z);
}

std::uint8_t StreamingWorld::water_level_at(
    const int global_x,
    const int global_y,
    const int global_z) const noexcept {
    if (!active_world_.has_value()) {
        return no_water_level;
    }
    return active_world_->water_level_at(
        global_x - active_region_.world_origin_x,
        global_y - active_region_.world_origin_y,
        global_z - active_region_.world_origin_z);
}

bool StreamingWorld::set_block(
    const int global_x,
    const int global_y,
    const int global_z,
    const BlockType block) {
    if (!active_world_.has_value()) {
        return false;
    }
    const std::unique_lock world_lock{active_world_mutex_};
    const int local_x = global_x - active_region_.world_origin_x;
    const int local_y = global_y - active_region_.world_origin_y;
    const int local_z = global_z - active_region_.world_origin_z;
    if (!active_world_->contains(local_x, local_y, local_z)) {
        return false;
    }

    active_world_->set_block(local_x, local_y, local_z, block);
    const VoxelPosition position{global_x, global_y, global_z};
    water_simulation_.notify_block_changed(*active_world_, {local_x, local_y, local_z});
    {
        const std::scoped_lock lock{edits_mutex_};
        edits_[position] = {
            .block = block,
            .water_level = block == BlockType::water ? water_source_level : no_water_level,
        };
        ++edit_revision_;
    }
    const auto dirty_chunks = rebuild_chunks_around_positions(
        *active_world_, active_region_, active_chunk_meshes_, {position});
    for (const std::size_t index : dirty_chunks) {
        if (std::find(dirty_chunk_indices_.begin(), dirty_chunk_indices_.end(), index) ==
            dirty_chunk_indices_.end()) {
            dirty_chunk_indices_.push_back(index);
        }
    }
    return true;
}

void StreamingWorld::advance_water(const float delta_seconds) {
    if (!active_world_.has_value()) {
        return;
    }
    const std::unique_lock world_lock{active_world_mutex_};
    const std::vector<VoxelPosition> local_changes =
        water_simulation_.advance(*active_world_, delta_seconds);
    if (local_changes.empty()) {
        return;
    }

    std::vector<VoxelPosition> global_changes;
    global_changes.reserve(local_changes.size());
    {
        const std::scoped_lock edits_lock{edits_mutex_};
        for (const VoxelPosition local : local_changes) {
            const VoxelPosition global{
                local.x + active_region_.world_origin_x,
                local.y + active_region_.world_origin_y,
                local.z + active_region_.world_origin_z,
            };
            global_changes.push_back(global);
            edits_[global] = {
                .block = BlockType::water,
                .water_level = active_world_->water_level_at(local.x, local.y, local.z),
            };
        }
        ++edit_revision_;
    }
    const auto rebuilt = rebuild_chunks_around_positions(
        *active_world_, active_region_, active_chunk_meshes_, global_changes);
    for (const std::size_t index : rebuilt) {
        if (std::find(dirty_chunk_indices_.begin(), dirty_chunk_indices_.end(), index) ==
            dirty_chunk_indices_.end()) {
            dirty_chunk_indices_.push_back(index);
        }
    }
}

const WorldMesh& StreamingWorld::chunk_mesh(const std::size_t index) const {
    return active_chunk_meshes_.at(index);
}

std::vector<std::size_t> StreamingWorld::consume_dirty_chunk_indices() {
    std::vector<std::size_t> dirty = std::move(dirty_chunk_indices_);
    dirty_chunk_indices_.clear();
    return dirty;
}

const std::vector<WorldMesh>& StreamingWorld::chunk_meshes() const noexcept {
    return active_chunk_meshes_;
}

std::vector<ChunkCoordinate> StreamingWorld::visible_chunk_coordinates() const {
    if (!center_chunk_.has_value()) {
        return {};
    }
    const int count = streaming_settings_.render_distance;
    std::vector<ChunkCoordinate> coordinates;
    coordinates.reserve(static_cast<std::size_t>(count * count));
    for (int chunk_z = 0; chunk_z < count; ++chunk_z) {
        for (int chunk_x = 0; chunk_x < count; ++chunk_x) {
            coordinates.push_back(window_coordinate(
                *center_chunk_, streaming_settings_, chunk_x, chunk_z));
        }
    }
    return coordinates;
}

std::optional<int> StreamingWorld::highest_solid_block(
    const int global_x,
    const int global_z) const noexcept {
    for (int y = maximum_y() - 1; y >= minimum_y(); --y) {
        if (is_solid(block_at(global_x, y, global_z))) {
            return y;
        }
    }
    return std::nullopt;
}

const std::optional<ChunkCoordinate>& StreamingWorld::center_chunk() const noexcept {
    return center_chunk_;
}

int StreamingWorld::minimum_y() const noexcept {
    return streaming_settings_.minimum_y;
}

int StreamingWorld::maximum_y() const noexcept {
    return streaming_settings_.minimum_y + streaming_settings_.world_height;
}

ChunkCoordinate StreamingWorld::chunk_at(const glm::vec3& camera_position) const noexcept {
    const float chunk_size = static_cast<float>(streaming_settings_.chunk_size);
    return {
        static_cast<int>(std::floor(camera_position.x / chunk_size)),
        static_cast<int>(std::floor(camera_position.z / chunk_size)),
    };
}

StreamingWorld::BuildResult StreamingWorld::rebuild(
    const ChunkCoordinate center,
    const std::optional<ChunkCoordinate> retained_center) const {
    const int chunk_size = streaming_settings_.chunk_size;
    const int visible_chunk_count = streaming_settings_.render_distance;
    const int full_detail_chunk_count = streaming_settings_.full_detail_radius * 2 + 1;
    const int full_detail_size = full_detail_chunk_count * chunk_size;
    const int padding = streaming_settings_.generation_padding;
    const int full_detail_origin_x =
        (center.x - streaming_settings_.full_detail_radius) * chunk_size;
    const int full_detail_origin_z =
        (center.z - streaming_settings_.full_detail_radius) * chunk_size;
    const int generated_origin_x = full_detail_origin_x - padding;
    const int generated_origin_z = full_detail_origin_z - padding;
    const MeshBuildRegion region{
        .minimum_x = padding,
        .maximum_x = padding + full_detail_size,
        .minimum_z = padding,
        .maximum_z = padding + full_detail_size,
        .world_origin_x = generated_origin_x,
        .world_origin_y = streaming_settings_.minimum_y,
        .world_origin_z = generated_origin_z,
    };

    WorldGenerator generator;
    const WorldDimensions generated_dimensions{
        full_detail_size + padding * 2,
        streaming_settings_.world_height,
        full_detail_size + padding * 2};
    World world = [&]() {
        const bool can_reuse = retained_center.has_value() && active_world_.has_value() &&
                               active_world_->dimensions().width == generated_dimensions.width &&
                               active_world_->dimensions().height == generated_dimensions.height &&
                               active_world_->dimensions().depth == generated_dimensions.depth;
        if (!can_reuse) {
            return generator.generate_region(
                generated_dimensions,
                generated_origin_x,
                generated_origin_z,
                generation_settings_,
                false);
        }

        const int old_origin_x = active_region_.world_origin_x;
        const int old_origin_z = active_region_.world_origin_z;
        const int new_maximum_x = generated_origin_x + generated_dimensions.width;
        const int new_maximum_z = generated_origin_z + generated_dimensions.depth;
        const int old_maximum_x = old_origin_x + active_world_->dimensions().width;
        const int old_maximum_z = old_origin_z + active_world_->dimensions().depth;
        const int overlap_minimum_x = std::max(generated_origin_x, old_origin_x);
        const int overlap_maximum_x = std::min(new_maximum_x, old_maximum_x);
        const int overlap_minimum_z = std::max(generated_origin_z, old_origin_z);
        const int overlap_maximum_z = std::min(new_maximum_z, old_maximum_z);
        if (overlap_minimum_x >= overlap_maximum_x ||
            overlap_minimum_z >= overlap_maximum_z) {
            return generator.generate_region(
                generated_dimensions,
                generated_origin_x,
                generated_origin_z,
                generation_settings_,
                false);
        }

        World destination{generated_dimensions};
        {
            const std::shared_lock world_lock{active_world_mutex_};
            destination.copy_columns_from(
                *active_world_,
                overlap_minimum_x - old_origin_x,
                overlap_minimum_z - old_origin_z,
                overlap_minimum_x - generated_origin_x,
                overlap_minimum_z - generated_origin_z,
                overlap_maximum_x - overlap_minimum_x,
                overlap_maximum_z - overlap_minimum_z);
        }
        const auto generate_rectangle = [&](
                                            const int minimum_x,
                                            const int maximum_x,
                                            const int minimum_z,
                                            const int maximum_z) {
            if (minimum_x >= maximum_x || minimum_z >= maximum_z) {
                return;
            }
            const int width = maximum_x - minimum_x;
            const int depth = maximum_z - minimum_z;
            World strip = generator.generate_region(
                {width + padding * 2,
                 streaming_settings_.world_height,
                 depth + padding * 2},
                minimum_x - padding,
                minimum_z - padding,
                generation_settings_,
                false);
            destination.copy_columns_from(
                strip,
                padding,
                padding,
                minimum_x - generated_origin_x,
                minimum_z - generated_origin_z,
                width,
                depth);
        };
        generate_rectangle(
            generated_origin_x,
            overlap_minimum_x,
            generated_origin_z,
            new_maximum_z);
        generate_rectangle(
            overlap_maximum_x,
            new_maximum_x,
            generated_origin_z,
            new_maximum_z);
        generate_rectangle(
            overlap_minimum_x,
            overlap_maximum_x,
            generated_origin_z,
            overlap_minimum_z);
        generate_rectangle(
            overlap_minimum_x,
            overlap_maximum_x,
            overlap_maximum_z,
            new_maximum_z);
        return destination;
    }();
    const std::uint64_t captured_edit_revision = apply_edits(world, region);
    world.rebuild_section_occupancy();
    auto chunk_meshes = build_chunk_meshes(world, region, center, retained_center);
    std::vector<std::size_t> upload_indices;
    std::vector<std::size_t> refreshed_chunk_indices;
    upload_indices.reserve(chunk_meshes.size());
    refreshed_chunk_indices.reserve(chunk_meshes.size());
    for (int chunk_z = 0; chunk_z < visible_chunk_count; ++chunk_z) {
        for (int chunk_x = 0; chunk_x < visible_chunk_count; ++chunk_x) {
            const ChunkCoordinate coordinate = window_coordinate(
                center, streaming_settings_, chunk_x, chunk_z);
            const bool retained = retained_center.has_value() &&
                                  contains_chunk(
                                      *retained_center, streaming_settings_, coordinate);
            if (!retained) {
                upload_indices.push_back(
                    static_cast<std::size_t>(chunk_z * visible_chunk_count + chunk_x));
            } else {
                const auto index =
                    static_cast<std::size_t>(chunk_z * visible_chunk_count + chunk_x);
                if (!chunk_meshes[index].vertices.empty()) {
                    refreshed_chunk_indices.push_back(index);
                }
            }
        }
    }
    upload_indices.insert(
        upload_indices.end(), refreshed_chunk_indices.begin(), refreshed_chunk_indices.end());
    WorldMesh mesh = combine_chunk_meshes(chunk_meshes, upload_indices);
    return {
        .world = std::move(world),
        .region = region,
        .chunk_meshes = std::move(chunk_meshes),
        .replacement_chunk_indices = std::move(upload_indices),
        .mesh = std::move(mesh),
        .center = center,
        .edit_revision = captured_edit_revision,
    };
}

void StreamingWorld::request_stream(const ChunkCoordinate center) {
    const std::optional<ChunkCoordinate> retained_center = center_chunk_;
    std::optional<World> retired_world = std::move(retired_world_);
    std::vector<WorldMesh> retired_chunk_meshes = std::move(retired_chunk_meshes_);
    std::optional<StreamedChunkUpdate> retired_update = std::move(retired_update_);
    pending_build_ = std::async(
        std::launch::async,
        [this,
         center,
         retained_center,
         retired_world = std::move(retired_world),
         retired_chunk_meshes = std::move(retired_chunk_meshes),
         retired_update = std::move(retired_update)]() mutable {
            retired_world.reset();
            retired_chunk_meshes.clear();
            retired_update.reset();
            return rebuild(center, retained_center);
        });
}

std::vector<WorldMesh> StreamingWorld::build_chunk_meshes(
    const World& world,
    const MeshBuildRegion& region,
    const ChunkCoordinate center,
    const std::optional<ChunkCoordinate> retained_center) const {
    const int count = streaming_settings_.render_distance;
    std::vector<WorldMesh> meshes(static_cast<std::size_t>(count * count));
    core::parallel_for(meshes.size(), [&](const std::size_t index) {
        const int chunk_x = static_cast<int>(index % static_cast<std::size_t>(count));
        const int chunk_z = static_cast<int>(index / static_cast<std::size_t>(count));
        const ChunkCoordinate coordinate = window_coordinate(
            center, streaming_settings_, chunk_x, chunk_z);
        const LodProfile profile = lod_profile(center, coordinate, streaming_settings_);
        if (retained_center.has_value() &&
            contains_chunk(*retained_center, streaming_settings_, coordinate)) {
            if (profile == lod_profile(*retained_center, coordinate, streaming_settings_)) {
                return;
            }
        }

        const WorldMesher mesher;
        if (!profile.surface) {
            const int minimum_x =
                (coordinate.x - (center.x - streaming_settings_.full_detail_radius)) *
                    streaming_settings_.chunk_size +
                region.minimum_x;
            const int minimum_z =
                (coordinate.z - (center.z - streaming_settings_.full_detail_radius)) *
                    streaming_settings_.chunk_size +
                region.minimum_z;
            meshes[index] = mesher.build_region(
                world,
                {
                    .minimum_x = minimum_x,
                    .maximum_x = minimum_x + streaming_settings_.chunk_size,
                    .minimum_z = minimum_z,
                    .maximum_z = minimum_z + streaming_settings_.chunk_size,
                    .world_origin_x = region.world_origin_x,
                    .world_origin_y = region.world_origin_y,
                    .world_origin_z = region.world_origin_z,
                });
            return;
        }

        const int padding = streaming_settings_.generation_padding * 2;
        const int compact_size = streaming_settings_.chunk_size + padding * 2;
        const int compact_origin_x = coordinate.x * streaming_settings_.chunk_size - padding;
        const int compact_origin_z = coordinate.z * streaming_settings_.chunk_size - padding;
        const WorldGenerator generator;
        World compact_source = generator.generate_region(
            {compact_size, streaming_settings_.world_height, compact_size},
            compact_origin_x,
            compact_origin_z,
            generation_settings_,
            false,
            false,
            true);
        meshes[index] = mesher.build_region(
            compact_source,
            {
                .minimum_x = padding,
                .maximum_x = padding + streaming_settings_.chunk_size,
                .minimum_z = padding,
                .maximum_z = padding + streaming_settings_.chunk_size,
                .world_origin_x = compact_origin_x,
                .world_origin_y = streaming_settings_.minimum_y,
                .world_origin_z = compact_origin_z,
                .surface_lod = true,
                .close_lod_minimum_x = profile.minimum_x,
                .close_lod_maximum_x = profile.maximum_x,
                .close_lod_minimum_z = profile.minimum_z,
                .close_lod_maximum_z = profile.maximum_z,
            });
    });
    return meshes;
}

WorldMesh StreamingWorld::combine_chunk_meshes(
    const std::vector<WorldMesh>& chunk_meshes,
    const std::vector<std::size_t>& selected_indices) {
    WorldMesh combined;
    std::size_t vertex_count = 0;
    std::size_t opaque_index_count = 0;
    std::size_t transparent_index_count = 0;
    std::size_t cloud_index_count = 0;
    std::size_t shadow_index_count = 0;
    for (const std::size_t selected_index : selected_indices) {
        const WorldMesh& mesh = chunk_meshes.at(selected_index);
        vertex_count += mesh.vertices.size();
        opaque_index_count += mesh.indices.size();
        transparent_index_count += mesh.transparent_indices.size();
        cloud_index_count += mesh.cloud_indices.size();
        shadow_index_count += mesh.shadow_indices.size();
    }
    combined.vertices.reserve(vertex_count);
    combined.indices.reserve(opaque_index_count);
    combined.transparent_indices.reserve(transparent_index_count);
    combined.cloud_indices.reserve(cloud_index_count);
    combined.shadow_indices.reserve(shadow_index_count);
    combined.draw_ranges.reserve(selected_indices.size());
    for (const std::size_t selected_index : selected_indices) {
        const WorldMesh& mesh = chunk_meshes.at(selected_index);
        const auto vertex_offset = static_cast<std::uint32_t>(combined.vertices.size());
        const MeshDrawRange range{
            .first_index = static_cast<std::uint32_t>(combined.indices.size()),
            .index_count = static_cast<std::uint32_t>(mesh.indices.size()),
            .first_transparent_index =
                static_cast<std::uint32_t>(combined.transparent_indices.size()),
            .transparent_index_count =
                static_cast<std::uint32_t>(mesh.transparent_indices.size()),
            .first_cloud_index = static_cast<std::uint32_t>(combined.cloud_indices.size()),
            .cloud_index_count = static_cast<std::uint32_t>(mesh.cloud_indices.size()),
            .first_shadow_index = static_cast<std::uint32_t>(combined.shadow_indices.size()),
            .shadow_index_count = static_cast<std::uint32_t>(mesh.shadow_indices.size()),
        };
        combined.vertices.insert(combined.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        for (const std::uint32_t element_index : mesh.indices) {
            combined.indices.push_back(vertex_offset + element_index);
        }
        for (const std::uint32_t element_index : mesh.transparent_indices) {
            combined.transparent_indices.push_back(vertex_offset + element_index);
        }
        for (const std::uint32_t element_index : mesh.cloud_indices) {
            combined.cloud_indices.push_back(vertex_offset + element_index);
        }
        for (const std::uint32_t element_index : mesh.shadow_indices) {
            combined.shadow_indices.push_back(vertex_offset + element_index);
        }
        combined.draw_ranges.push_back(range);
    }
    return combined;
}

std::vector<std::size_t> StreamingWorld::rebuild_chunks_around_positions(
    const World& world,
    const MeshBuildRegion& region,
    std::vector<WorldMesh>& chunk_meshes,
    const std::vector<VoxelPosition>& positions) const {
    const int count = streaming_settings_.render_distance;
    if (chunk_meshes.size() != static_cast<std::size_t>(count * count)) {
        chunk_meshes = build_chunk_meshes(
            world,
            region,
            center_chunk_.value_or(ChunkCoordinate{}),
            std::nullopt);
        std::vector<std::size_t> all_chunks(chunk_meshes.size());
        std::iota(all_chunks.begin(), all_chunks.end(), 0U);
        return all_chunks;
    }

    const ChunkCoordinate center = center_chunk_.value_or(ChunkCoordinate{});
    const int minimum_offset = minimum_window_offset(streaming_settings_);
    std::vector<bool> dirty(chunk_meshes.size(), false);
    for (const VoxelPosition& position : positions) {
        const int position_chunk_x = floor_divide(position.x, streaming_settings_.chunk_size);
        const int position_chunk_z = floor_divide(position.z, streaming_settings_.chunk_size);
        for (int candidate_z = position_chunk_z - 1; candidate_z <= position_chunk_z + 1;
             ++candidate_z) {
            for (int candidate_x = position_chunk_x - 1; candidate_x <= position_chunk_x + 1;
                 ++candidate_x) {
                const ChunkCoordinate coordinate{candidate_x, candidate_z};
                if (!contains_chunk(center, streaming_settings_, coordinate) ||
                    uses_surface_lod(center, coordinate, streaming_settings_)) {
                    continue;
                }
                const int minimum_x = candidate_x * streaming_settings_.chunk_size;
                const int minimum_z = candidate_z * streaming_settings_.chunk_size;
                const int maximum_x = minimum_x + streaming_settings_.chunk_size;
                const int maximum_z = minimum_z + streaming_settings_.chunk_size;
                if (position.x >= minimum_x - 1 && position.x <= maximum_x &&
                    position.z >= minimum_z - 1 && position.z <= maximum_z) {
                    const int slot_x = candidate_x - (center.x + minimum_offset);
                    const int slot_z = candidate_z - (center.z + minimum_offset);
                    dirty[static_cast<std::size_t>(slot_z * count + slot_x)] = true;
                }
            }
        }
    }

    WorldMesher mesher;
    std::vector<std::size_t> rebuilt;
    for (int chunk_z = 0; chunk_z < count; ++chunk_z) {
        for (int chunk_x = 0; chunk_x < count; ++chunk_x) {
            const auto index = static_cast<std::size_t>(chunk_z * count + chunk_x);
            if (!dirty[index]) {
                continue;
            }
            const ChunkCoordinate coordinate = window_coordinate(
                center, streaming_settings_, chunk_x, chunk_z);
            const int minimum_x =
                (coordinate.x - (center.x - streaming_settings_.full_detail_radius)) *
                    streaming_settings_.chunk_size +
                region.minimum_x;
            const int minimum_z =
                (coordinate.z - (center.z - streaming_settings_.full_detail_radius)) *
                    streaming_settings_.chunk_size +
                region.minimum_z;
            chunk_meshes[index] = mesher.build_region(
                world,
                {
                    .minimum_x = minimum_x,
                    .maximum_x = minimum_x + streaming_settings_.chunk_size,
                    .minimum_z = minimum_z,
                    .maximum_z = minimum_z + streaming_settings_.chunk_size,
                    .world_origin_x = region.world_origin_x,
                    .world_origin_y = region.world_origin_y,
                    .world_origin_z = region.world_origin_z,
                });
            rebuilt.push_back(index);
        }
    }
    return rebuilt;
}

std::uint64_t StreamingWorld::apply_edits(World& world, const MeshBuildRegion& region) const {
    const std::scoped_lock lock{edits_mutex_};
    for (const auto& [position, edit] : edits_) {
        const int local_x = position.x - region.world_origin_x;
        const int local_y = position.y - region.world_origin_y;
        const int local_z = position.z - region.world_origin_z;
        if (world.contains(local_x, local_y, local_z)) {
            if (edit.block == BlockType::water) {
                world.set_water(local_x, local_y, local_z, edit.water_level);
            } else {
                world.set_block_untracked(local_x, local_y, local_z, edit.block);
            }
        }
    }
    return edit_revision_;
}

std::uint64_t StreamingWorld::edit_revision() const {
    const std::scoped_lock lock{edits_mutex_};
    return edit_revision_;
}

}  // namespace vulkancraft::world
