#pragma once

#include <glm/vec3.hpp>

namespace vulkancraft::world {
class StreamingWorld;
}

namespace vulkancraft::core {

/** Describes movement controls sampled for one simulation step. */
struct PlayerInput {
    float forward{};
    float right{};
    bool jump{};
    bool sprint{};
    bool crouch{};
};

/** Simulates a first-person capsule-like voxel character. */
class PlayerController final {
public:
    /** Places the player safely above the generated terrain. */
    void spawn(const world::StreamingWorld& world, float x, float z) noexcept;

    /** Advances collision, gravity, jumping, and swimming. */
    void update(
        const PlayerInput& input,
        const glm::vec3& view_forward,
        float delta_seconds,
        const world::StreamingWorld& world) noexcept;

    /** Teleports the player while retaining camera orientation. */
    void set_position(glm::vec3 feet_position) noexcept;

    /** Returns the current feet position. */
    [[nodiscard]] const glm::vec3& position() const noexcept;

    /** Returns the current camera position, including crouch eye height. */
    [[nodiscard]] glm::vec3 eye_position() const noexcept;

    /** Reports whether a target voxel would overlap the player body. */
    [[nodiscard]] bool occupies_block(int x, int y, int z) const noexcept;

    /** Reports whether the player's head or torso is submerged. */
    [[nodiscard]] bool is_swimming() const noexcept;

    /** Reports whether double-tap flight is currently enabled. */
    [[nodiscard]] bool is_flying() const noexcept;

private:
    /** Tests the player's axis-aligned body against collidable voxels. */
    [[nodiscard]] bool collides(
        const glm::vec3& position,
        float height,
        const world::StreamingWorld& world) const noexcept;

    glm::vec3 position_{0.5F, 80.0F, 0.5F};
    glm::vec3 velocity_{};
    bool grounded_{false};
    bool swimming_{false};
    bool crouching_{false};
    bool flying_{false};
    bool jump_was_pressed_{false};
    bool waiting_for_second_jump_tap_{false};
    float jump_tap_elapsed_seconds_{};
};

}  // namespace vulkancraft::core
