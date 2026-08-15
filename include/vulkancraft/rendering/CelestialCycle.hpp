#pragma once

#include <glm/vec3.hpp>

namespace vulkancraft::rendering {

/** Duration of the sunlit half of one world day, in seconds. */
inline constexpr float daylight_duration_seconds = 12.0F * 60.0F;

/** Duration of one complete day-night cycle, in seconds. */
inline constexpr float full_day_duration_seconds = daylight_duration_seconds * 2.0F;

/** Defines the world-fixed axes used to orient one celestial quad. */
struct CelestialBasis {
    glm::vec3 right{};
    glm::vec3 up{};
};

/**
 * Calculates the centered direction of the sun during the day-night cycle.
 *
 * The orbit lies in the vertical Y-Z plane. Angle zero is sunrise at the
 * center of the default forward horizon, and pi radians is sunset at the
 * opposite horizon.
 *
 * @param day_angle Current orbital angle in radians.
 * @return Normalized world-space direction from the observer to the sun.
 */
[[nodiscard]] glm::vec3 celestial_direction(float day_angle) noexcept;

/**
 * Converts elapsed world time into the centered celestial orbit angle.
 *
 * @param elapsed_seconds Accumulated world-clock seconds.
 * @return Wrapped orbit angle in radians, where zero is sunrise.
 */
[[nodiscard]] float celestial_angle(float elapsed_seconds) noexcept;

/**
 * Calculates a camera-independent basis for one celestial direction.
 *
 * @param direction Normalized direction from the observer to the celestial body.
 * @return World-fixed right and up axes for the quad.
 */
[[nodiscard]] CelestialBasis celestial_basis(const glm::vec3& direction) noexcept;

}  // namespace vulkancraft::rendering
