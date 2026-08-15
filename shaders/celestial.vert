#version 450

layout(set = 0, binding = 0) uniform FrameUniforms {
    mat4 view;
    mat4 projection;
    mat4 lightViewProjection[4];
    vec4 cascadeSplits;
    vec4 lightDirection;
    vec4 cameraPosition;
    vec4 lightColorIntensity;
    vec4 skyColor;
    vec4 sunDirection;
} frame;

layout(push_constant) uniform CelestialPushConstants {
    vec4 directionAndHalfSize;
    vec4 color;
    vec4 worldRight;
    vec4 worldUp;
} celestial;

void main() {
    const vec2 corners[6] = vec2[6](
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    vec3 direction = normalize(celestial.directionAndHalfSize.xyz);
    vec2 corner = corners[gl_VertexIndex] * celestial.directionAndHalfSize.w;
    vec3 center = frame.cameraPosition.xyz + direction * 150.0;
    vec3 worldPosition = center + celestial.worldRight.xyz * corner.x +
                         celestial.worldUp.xyz * corner.y;
    gl_Position = frame.projection * frame.view * vec4(worldPosition, 1.0);
}
