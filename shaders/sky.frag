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

layout(location = 0) in vec2 inClipPosition;
layout(location = 0) out vec4 outColor;

void main() {
    vec4 viewPosition = inverse(frame.projection) * vec4(inClipPosition, 1.0, 1.0);
    vec3 viewRay = normalize(viewPosition.xyz / viewPosition.w);
    vec3 worldRay = normalize((inverse(frame.view) * vec4(viewRay, 0.0)).xyz);
    float horizon = clamp(worldRay.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky = mix(frame.skyColor.rgb * 0.58, frame.skyColor.rgb, horizon);
    if (frame.cameraPosition.w > 0.5) {
        sky = vec3(0.008, 0.055, 0.15);
    }
    outColor = vec4(sky, 1.0);
}
