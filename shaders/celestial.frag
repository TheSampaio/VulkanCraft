#version 450

layout(push_constant) uniform CelestialPushConstants {
    vec4 directionAndHalfSize;
    vec4 color;
    vec4 worldRight;
    vec4 worldUp;
} celestial;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = celestial.color;
}
