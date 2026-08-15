#pragma once

#include <cstdint>
#include <future>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include <glm/vec3.hpp>

#include "vulkancraft/world/WorldGenerator.hpp"
#include "vulkancraft/world/WorldMesher.hpp"
#include "vulkancraft/world/WaterSimulation.hpp"

namespace vulkancraft::world {

/** Identifies one horizontal chunk in the unbounded world grid. */
struct ChunkCoordinate {
    int x{};
    int z{};

    bool operator==(const ChunkCoordinate&) const = default;
};

/** Controls the generated window retained around the player. */
struct WorldStreamingSettings {
    int chunk_size{32};
    int render_distance{32};
    int full_detail_radius{8};
    int world_height{384};
    int minimum_y{-64};
    int generation_padding{3};
};

/** Owns one asynchronously completed GPU chunk-window replacement batch. */
struct StreamedChunkUpdate {
    WorldMesh mesh;
    std::vector<std::size_t> replacement_indices;
};

/** Maintains an asynchronously generated window with cached per-chunk meshes. */
class StreamingWorld final {
public:
    /** Creates a validated streaming world configuration. */
    explicit StreamingWorld(
        WorldGenerationSettings generation = {},
        WorldStreamingSettings streaming = {});

    /** Waits for an outstanding worker before releasing shared generation state. */
    ~StreamingWorld();

    /** Generates the first visible window and returns its batched mesh. */
    [[nodiscard]] WorldMesh initialize(const glm::vec3& camera_position);

    /** Polls background streaming and returns one packed replacement batch when ready. */
    [[nodiscard]] std::optional<StreamedChunkUpdate> update(
        const glm::vec3& camera_position);

    /**
     * Defers release of an uploaded CPU replacement batch to the next background rebuild.
     *
     * @param update Replacement data already copied into renderer-owned GPU storage.
     */
    void retire_update(StreamedChunkUpdate&& update) noexcept;

    /** Reads a global voxel, treating unloaded positions as air. */
    [[nodiscard]] BlockType block_at(int global_x, int global_y, int global_z) const noexcept;

    /** Replaces one loaded voxel and remeshes only neighboring cached chunks. */
    [[nodiscard]] bool set_block(int global_x, int global_y, int global_z, BlockType block);

    /**
     * Advances scheduled water and queues partial GPU chunk replacements.
     *
     * @param delta_seconds Elapsed real time since the previous frame.
     */
    void advance_water(float delta_seconds);

    /**
     * Reads source, horizontal-flow, or falling metadata for a loaded water voxel.
     *
     * @param global_x Global horizontal x coordinate.
     * @param global_y Global vertical coordinate.
     * @param global_z Global horizontal z coordinate.
     * @return Fluid level or `no_water_level` when the position is not loaded water.
     */
    [[nodiscard]] std::uint8_t water_level_at(
        int global_x,
        int global_y,
        int global_z) const noexcept;

    /** Returns one cached chunk mesh for a renderer-side partial update. */
    [[nodiscard]] const WorldMesh& chunk_mesh(std::size_t index) const;

    /** Moves out the cached chunk indices changed by recent block edits. */
    [[nodiscard]] std::vector<std::size_t> consume_dirty_chunk_indices();

    /** Returns all active cached meshes in stable row-major chunk order. */
    [[nodiscard]] const std::vector<WorldMesh>& chunk_meshes() const noexcept;

    /** Returns global coordinates for the active row-major chunk meshes. */
    [[nodiscard]] std::vector<ChunkCoordinate> visible_chunk_coordinates() const;

    /** Finds the highest collidable block in a loaded column. */
    [[nodiscard]] std::optional<int> highest_solid_block(int global_x, int global_z) const noexcept;

    /** Returns the chunk currently represented by the active voxel window. */
    [[nodiscard]] const std::optional<ChunkCoordinate>& center_chunk() const noexcept;

    /** Returns the configured lower inclusive world limit. */
    [[nodiscard]] int minimum_y() const noexcept;

    /** Returns the configured upper exclusive world limit. */
    [[nodiscard]] int maximum_y() const noexcept;

private:
    struct BlockEdit {
        BlockType block{BlockType::air};
        std::uint8_t water_level{no_water_level};
    };

    struct BuildResult {
        std::optional<World> world;
        MeshBuildRegion region;
        std::vector<WorldMesh> chunk_meshes;
        std::vector<std::size_t> replacement_chunk_indices;
        WorldMesh mesh;
        ChunkCoordinate center;
        std::uint64_t edit_revision{};
    };

    /** Converts a floating-point position to a floor-divided chunk coordinate. */
    [[nodiscard]] ChunkCoordinate chunk_at(const glm::vec3& camera_position) const noexcept;

    /** Generates a window and precombines only chunks absent from the retained window. */
    [[nodiscard]] BuildResult rebuild(
        ChunkCoordinate center,
        std::optional<ChunkCoordinate> retained_center) const;

    /** Starts a background window generation request. */
    void request_stream(ChunkCoordinate center);

    /** Builds one independent mesh for each visible horizontal chunk. */
    [[nodiscard]] std::vector<WorldMesh> build_chunk_meshes(
        const World& world,
        const MeshBuildRegion& region,
        ChunkCoordinate center,
        std::optional<ChunkCoordinate> retained_center) const;

    /** Combines selected cached chunk meshes into renderer draw batches. */
    [[nodiscard]] static WorldMesh combine_chunk_meshes(
        const std::vector<WorldMesh>& chunk_meshes,
        const std::vector<std::size_t>& selected_indices);

    /** Rebuilds chunks touched by edited voxels, including shared boundaries. */
    [[nodiscard]] std::vector<std::size_t> rebuild_chunks_around_positions(
        const World& world,
        const MeshBuildRegion& region,
        std::vector<WorldMesh>& chunk_meshes,
        const std::vector<VoxelPosition>& positions) const;

    /** Applies session edits and returns the captured edit revision. */
    [[nodiscard]] std::uint64_t apply_edits(World& world, const MeshBuildRegion& region) const;

    /** Returns the current thread-safe session edit revision. */
    [[nodiscard]] std::uint64_t edit_revision() const;

    WorldGenerationSettings generation_settings_;
    WorldStreamingSettings streaming_settings_;
    std::optional<World> active_world_;
    std::optional<World> retired_world_;
    std::vector<WorldMesh> retired_chunk_meshes_;
    std::optional<StreamedChunkUpdate> retired_update_;
    MeshBuildRegion active_region_{};
    std::vector<WorldMesh> active_chunk_meshes_;
    std::vector<std::size_t> dirty_chunk_indices_;
    std::optional<ChunkCoordinate> center_chunk_;
    std::future<BuildResult> pending_build_;
    WaterSimulation water_simulation_;
    mutable std::shared_mutex active_world_mutex_;
    mutable std::mutex edits_mutex_;
    std::unordered_map<VoxelPosition, BlockEdit, VoxelPositionHash> edits_;
    std::uint64_t edit_revision_{};
};

}  // namespace vulkancraft::world
