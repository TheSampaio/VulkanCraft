#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTextureCoordinate;
layout(location = 2) in uint inAttributes;

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
    vec4 animationData;
} frame;

layout(push_constant) uniform CascadeIndex {
    int value;
} cascade;

layout(location = 0) out vec2 outTextureCoordinate;
layout(location = 1) flat out uint outMaterial;

void main() {
    const vec2 windDirection = normalize(vec2(0.92, 0.38));
    const uint material = (inAttributes >> 3u) & 0x3u;
    const uint windClass = (inAttributes >> 5u) & 0x3u;
    vec3 animatedPosition = inPosition;
    if (material == 3u) {
        const float animationSpeed = 0.055 + float(windClass) * 0.025;
        animatedPosition.y += float(windClass) * 2.5;
        animatedPosition.xz += windDirection * frame.animationData.x * animationSpeed;
    }
    gl_Position = frame.lightViewProjection[cascade.value] * vec4(animatedPosition, 1.0);
    outTextureCoordinate = inTextureCoordinate;
    outMaterial = material;
}
