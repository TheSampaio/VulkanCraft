#include "vulkancraft/core/PlayerController.hpp"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>

#include "vulkancraft/world/StreamingWorld.hpp"

namespace vulkancraft::core {
namespace {

constexpr float player_radius = 0.30F;
constexpr float standing_height = 1.80F;
constexpr float crouching_height = 1.50F;
constexpr float standing_eye_height = 1.62F;
constexpr float crouching_eye_height = 1.27F;
constexpr float double_tap_window_seconds = 0.30F;

}  // namespace

void PlayerController::spawn(
    const world::StreamingWorld& world,
    const float x,
    const float z) noexcept {
    const int origin_x = static_cast<int>(std::floor(x));
    const int origin_z = static_cast<int>(std::floor(z));
    int spawn_x = origin_x;
    int spawn_z = origin_z;
    int ground = world.highest_solid_block(origin_x, origin_z).value_or(70);
    bool found_dry_ground = false;
    for (int radius = 0; radius <= 64 && !found_dry_ground; ++radius) {
        for (int offset_z = -radius; offset_z <= radius && !found_dry_ground; ++offset_z) {
            for (int offset_x = -radius; offset_x <= radius; ++offset_x) {
                if (radius > 0 && std::abs(offset_x) != radius && std::abs(offset_z) != radius) {
                    continue;
                }
                const int candidate_x = origin_x + offset_x;
                const int candidate_z = origin_z + offset_z;
                const auto candidate_ground = world.highest_solid_block(candidate_x, candidate_z);
                if (!candidate_ground.has_value() ||
                    world.block_at(candidate_x, *candidate_ground + 1, candidate_z) ==
                        world::BlockType::water) {
                    continue;
                }
                spawn_x = candidate_x;
                spawn_z = candidate_z;
                ground = *candidate_ground;
                found_dry_ground = true;
                break;
            }
        }
    }
    position_ = {
        static_cast<float>(spawn_x) + 0.5F,
        static_cast<float>(ground + 1) + 0.01F,
        static_cast<float>(spawn_z) + 0.5F};
    velocity_ = {};
    flying_ = false;
    jump_was_pressed_ = false;
    waiting_for_second_jump_tap_ = false;
    jump_tap_elapsed_seconds_ = 0.0F;
}

void PlayerController::update(
    const PlayerInput& input,
    const glm::vec3& view_forward,
    const float delta_seconds,
    const world::StreamingWorld& world) noexcept {
    const float step_seconds = std::clamp(delta_seconds, 0.0F, 0.05F);
    crouching_ = input.crouch;
    const float body_height = crouching_ ? crouching_height : standing_height;
    jump_tap_elapsed_seconds_ += step_seconds;
    const bool jump_pressed = input.jump && !jump_was_pressed_;
    jump_was_pressed_ = input.jump;
    if (jump_pressed) {
        if (waiting_for_second_jump_tap_ &&
            jump_tap_elapsed_seconds_ <= double_tap_window_seconds) {
            flying_ = !flying_;
            velocity_.y = 0.0F;
            waiting_for_second_jump_tap_ = false;
        } else {
            waiting_for_second_jump_tap_ = true;
            jump_tap_elapsed_seconds_ = 0.0F;
        }
    } else if (waiting_for_second_jump_tap_ &&
               jump_tap_elapsed_seconds_ > double_tap_window_seconds) {
        waiting_for_second_jump_tap_ = false;
    }

    glm::vec3 forward{view_forward.x, 0.0F, view_forward.z};
    if (glm::length(forward) < 0.001F) {
        forward = {0.0F, 0.0F, -1.0F};
    } else {
        forward = glm::normalize(forward);
    }
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3{0.0F, 1.0F, 0.0F}));
    glm::vec3 wish = forward * input.forward + right * input.right;
    if (glm::length(wish) > 1.0F) {
        wish = glm::normalize(wish);
    }

    const auto block_at_height = [&world, this](const float height) {
        return world.block_at(
            static_cast<int>(std::floor(position_.x)),
            static_cast<int>(std::floor(position_.y + height)),
            static_cast<int>(std::floor(position_.z)));
    };
    swimming_ = !flying_ && (block_at_height(0.25F) == world::BlockType::water ||
                             block_at_height(1.25F) == world::BlockType::water);

    float movement_speed = flying_ ? (input.sprint ? 28.0F : 11.0F)
                                    : (input.sprint ? 8.2F : 4.7F);
    if (crouching_) {
        movement_speed *= 0.45F;
    }
    if (flying_) {
        const float vertical_intent = (input.jump ? 1.0F : 0.0F) -
                                      (input.crouch ? 1.0F : 0.0F);
        velocity_.y = vertical_intent * (input.sprint ? 24.0F : 11.0F);
        grounded_ = false;
    } else if (swimming_) {
        movement_speed *= 0.58F;
    }
    velocity_.x = wish.x * movement_speed;
    velocity_.z = wish.z * movement_speed;

    if (!flying_) {
        if (swimming_) {
            velocity_.y = std::max(velocity_.y - 4.0F * step_seconds, -3.0F);
            if (input.jump) {
                velocity_.y = std::min(velocity_.y + 13.0F * step_seconds, 3.5F);
            }
        } else {
            if (jump_pressed && grounded_) {
                velocity_.y = 8.1F;
                grounded_ = false;
            }
            velocity_.y = std::max(velocity_.y - 24.0F * step_seconds, -42.0F);
        }
    }

    const glm::vec3 total_motion = velocity_ * step_seconds;
    const int substeps = std::max(1, static_cast<int>(std::ceil(glm::length(total_motion) / 0.20F)));
    const glm::vec3 motion = total_motion / static_cast<float>(substeps);
    grounded_ = false;
    for (int step = 0; step < substeps; ++step) {
        glm::vec3 candidate = position_;
        candidate.x += motion.x;
        if (!collides(candidate, body_height, world)) {
            position_.x = candidate.x;
        } else {
            velocity_.x = 0.0F;
        }

        candidate = position_;
        candidate.z += motion.z;
        if (!collides(candidate, body_height, world)) {
            position_.z = candidate.z;
        } else {
            velocity_.z = 0.0F;
        }

        candidate = position_;
        candidate.y += motion.y;
        if (!collides(candidate, body_height, world)) {
            position_.y = candidate.y;
        } else {
            if (motion.y < 0.0F) {
                grounded_ = true;
            }
            velocity_.y = 0.0F;
        }
    }
}

void PlayerController::set_position(const glm::vec3 feet_position) noexcept {
    position_ = feet_position;
    velocity_ = {};
}

const glm::vec3& PlayerController::position() const noexcept {
    return position_;
}

glm::vec3 PlayerController::eye_position() const noexcept {
    return position_ + glm::vec3{0.0F, crouching_ ? crouching_eye_height : standing_eye_height, 0.0F};
}

bool PlayerController::occupies_block(const int x, const int y, const int z) const noexcept {
    const float height = crouching_ ? crouching_height : standing_height;
    return static_cast<float>(x + 1) > position_.x - player_radius &&
           static_cast<float>(x) < position_.x + player_radius &&
           static_cast<float>(y + 1) > position_.y &&
           static_cast<float>(y) < position_.y + height &&
           static_cast<float>(z + 1) > position_.z - player_radius &&
           static_cast<float>(z) < position_.z + player_radius;
}

bool PlayerController::is_swimming() const noexcept {
    return swimming_;
}

bool PlayerController::is_flying() const noexcept {
    return flying_;
}

bool PlayerController::collides(
    const glm::vec3& position,
    const float height,
    const world::StreamingWorld& world) const noexcept {
    constexpr float epsilon = 0.001F;
    const int minimum_x = static_cast<int>(std::floor(position.x - player_radius + epsilon));
    const int maximum_x = static_cast<int>(std::floor(position.x + player_radius - epsilon));
    const int minimum_y = static_cast<int>(std::floor(position.y + epsilon));
    const int maximum_y = static_cast<int>(std::floor(position.y + height - epsilon));
    const int minimum_z = static_cast<int>(std::floor(position.z - player_radius + epsilon));
    const int maximum_z = static_cast<int>(std::floor(position.z + player_radius - epsilon));
    for (int y = minimum_y; y <= maximum_y; ++y) {
        for (int z = minimum_z; z <= maximum_z; ++z) {
            for (int x = minimum_x; x <= maximum_x; ++x) {
                if (world::is_solid(world.block_at(x, y, z))) {
                    return true;
                }
            }
        }
    }
    return false;
}

}  // namespace vulkancraft::core
