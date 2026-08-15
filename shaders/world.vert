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

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) flat out uint outMaterial;
layout(location = 3) out vec2 outTextureCoordinate;
layout(location = 4) out float outViewDepth;
layout(location = 5) out vec4 outShadowPosition[4];

vec3 decodeNormal(uint normalIndex) {
    const vec3 normals[6] = vec3[6](
        vec3(1.0, 0.0, 0.0),
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0));
    return normals[min(normalIndex, 5u)];
}

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
    vec4 worldPosition = vec4(animatedPosition, 1.0);
    vec4 viewPosition = frame.view * worldPosition;
    gl_Position = frame.projection * viewPosition;
    outWorldPosition = animatedPosition;
    outNormal = decodeNormal(inAttributes & 0x7u);
    outMaterial = material;
    outTextureCoordinate = inTextureCoordinate;
    outViewDepth = -viewPosition.z;
    for (int cascade = 0; cascade < 4; ++cascade) {
        outShadowPosition[cascade] = frame.lightViewProjection[cascade] * worldPosition;
    }
}
