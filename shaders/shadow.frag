#version 450

layout(set = 0, binding = 2) uniform sampler2D blockAtlas;

layout(location = 0) in vec2 inTextureCoordinate;
layout(location = 1) flat in uint inMaterial;

void main() {
    if (inMaterial != 1u) {
        return;
    }
    const ivec2 atlasSize = textureSize(blockAtlas, 0);
    const ivec2 atlasTexel = clamp(
        ivec2(floor(inTextureCoordinate * vec2(atlasSize))),
        ivec2(0),
        atlasSize - ivec2(1));
    if (texelFetch(blockAtlas, atlasTexel, 0).a < 0.1) {
        discard;
    }
}
