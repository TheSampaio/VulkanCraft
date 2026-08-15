#version 450

layout(set = 0, binding = 2) uniform sampler2D blockAtlas;

layout(location = 0) in vec2 inTextureCoordinate;

const int atlasMaximumMip = 5;

int selectAtlasMip() {
    vec2 baseTexelCoordinate = inTextureCoordinate * vec2(textureSize(blockAtlas, 0));
    float footprint = max(length(dFdx(baseTexelCoordinate)), length(dFdy(baseTexelCoordinate)));
    return clamp(int(floor(log2(max(footprint, 1.0)))), 0, atlasMaximumMip);
}

void main() {
    int atlasMip = selectAtlasMip();
    ivec2 atlasSize = textureSize(blockAtlas, atlasMip);
    ivec2 atlasTexel = clamp(
        ivec2(floor(inTextureCoordinate * vec2(atlasSize))),
        ivec2(0),
        atlasSize - ivec2(1));
    if (texelFetch(blockAtlas, atlasTexel, atlasMip).a < 0.1) {
        discard;
    }
}
