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
    vec4 animationData;
} frame;

layout(set = 0, binding = 1) uniform sampler2DArray shadowMap;
layout(set = 0, binding = 2) uniform sampler2D blockAtlas;

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) flat in uint inMaterial;
layout(location = 3) in vec2 inTextureCoordinate;
layout(location = 4) in float inViewDepth;
layout(location = 5) in vec4 inShadowPosition[4];

layout(location = 0) out vec4 outColor;

const int atlasMaximumMip = 5;

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 waterWaveNormal(vec3 worldPosition) {
    const vec2 firstDirection = normalize(vec2(0.894, 0.447));
    const vec2 secondDirection = normalize(vec2(-0.351, 0.936));
    const float firstPhase = dot(worldPosition.xz, firstDirection) * 0.42 +
                             frame.animationData.x * 0.55;
    const float secondPhase = dot(worldPosition.xz, secondDirection) * 0.27 -
                              frame.animationData.x * 0.37;
    const vec2 slope = firstDirection * cos(firstPhase) * 0.065 +
                       secondDirection * cos(secondPhase) * 0.035;
    return normalize(vec3(-slope.x, 1.0, -slope.y));
}

int selectAtlasMip() {
    vec2 baseTexelCoordinate = inTextureCoordinate * vec2(textureSize(blockAtlas, 0));
    float footprint = max(length(dFdx(baseTexelCoordinate)), length(dFdy(baseTexelCoordinate)));
    return clamp(int(floor(log2(max(footprint, 1.0)))), 0, atlasMaximumMip);
}

float sampleShadow(int cascade, vec3 normal, vec3 toLight) {
    vec3 projected = inShadowPosition[cascade].xyz / inShadowPosition[cascade].w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }

    float normalBias = max(0.0011 * (1.0 - dot(normal, toLight)), 0.00028);
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0).xy);
    float visibility = 0.0;
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const float storedDepth = texture(
                shadowMap,
                vec3(uv + vec2(offsetX, offsetY) * texel, float(cascade))).r;
            visibility += projected.z - normalBias <= storedDepth ? 1.0 : 0.0;
        }
    }
    return visibility / 9.0;
}

void main() {
    int cascade = 0;
    if (inViewDepth > frame.cascadeSplits.x) cascade = 1;
    if (inViewDepth > frame.cascadeSplits.y) cascade = 2;
    if (inViewDepth > frame.cascadeSplits.z) cascade = 3;

    vec3 normal = normalize(inNormal);
    vec3 toLight = normalize(-frame.lightDirection.xyz);
    float diffuse = max(dot(normal, toLight), 0.0);
    bool leavesMaterial = inMaterial == 1u;
    bool waterMaterial = inMaterial == 2u;
    bool cloudMaterial = inMaterial == 3u;
    bool blendedMaterial = waterMaterial || cloudMaterial;
    if (waterMaterial && normal.y > 0.72) {
        normal = waterWaveNormal(inWorldPosition);
    }
    float visibility = cloudMaterial || diffuse <= 0.001 ||
                               inViewDepth > frame.cascadeSplits.w
                           ? 1.0
                           : sampleShadow(cascade, normal, toLight);
    int atlasMip = selectAtlasMip();
    ivec2 atlasSize = textureSize(blockAtlas, atlasMip);
    ivec2 atlasTexel = clamp(
        ivec2(floor(inTextureCoordinate * vec2(atlasSize))),
        ivec2(0),
        atlasSize - ivec2(1));
    vec4 texel = texelFetch(blockAtlas, atlasTexel, atlasMip);
    if (waterMaterial) {
        ivec2 waterOrigin = ivec2(32, 64) >> atlasMip;
        int mipCellSize = max(1, 32 >> atlasMip);
        ivec2 cellSize = ivec2(mipCellSize);
        const ivec2 localTexel = clamp(atlasTexel - waterOrigin, ivec2(0), cellSize - ivec2(1));
        const int animationStep = int(floor(frame.animationData.x * 2.0));
        const ivec2 offset = ivec2(animationStep % mipCellSize, (animationStep / 2) % mipCellSize);
        const ivec2 wrappedTexel = (localTexel + offset) % cellSize;
        const bool horizontalEdge = localTexel.x == 0 || localTexel.x == mipCellSize - 1;
        const bool verticalEdge = localTexel.y == 0 || localTexel.y == mipCellSize - 1;
        const int firstX = horizontalEdge ? offset.x : wrappedTexel.x;
        const int secondX = horizontalEdge ? (offset.x + mipCellSize - 1) % mipCellSize
                                           : wrappedTexel.x;
        const int firstY = verticalEdge ? offset.y : wrappedTexel.y;
        const int secondY = verticalEdge ? (offset.y + mipCellSize - 1) % mipCellSize
                                         : wrappedTexel.y;
        const float sampleCount = float((horizontalEdge ? 2 : 1) * (verticalEdge ? 2 : 1));
        texel = texelFetch(blockAtlas, waterOrigin + ivec2(firstX, firstY), atlasMip);
        if (horizontalEdge) {
            texel += texelFetch(blockAtlas, waterOrigin + ivec2(secondX, firstY), atlasMip);
        }
        if (verticalEdge) {
            texel += texelFetch(blockAtlas, waterOrigin + ivec2(firstX, secondY), atlasMip);
        }
        if (horizontalEdge && verticalEdge) {
            texel += texelFetch(blockAtlas, waterOrigin + ivec2(secondX, secondY), atlasMip);
        }
        texel /= sampleCount;
    }
    if (!blendedMaterial && texel.a < 0.1) discard;
    const int materialCellSize = max(1, 32 >> atlasMip);
    const ivec2 materialCellOrigin = waterMaterial
        ? ivec2(32, 64) >> atlasMip
        : (atlasTexel / materialCellSize) * materialCellSize;
    const ivec2 materialLocalTexel = clamp(
        atlasTexel - materialCellOrigin,
        ivec2(0),
        ivec2(materialCellSize - 1));
    const ivec2 neighborX = materialCellOrigin + ivec2(
        min(materialLocalTexel.x + 1, materialCellSize - 1), materialLocalTexel.y);
    const ivec2 neighborY = materialCellOrigin + ivec2(
        materialLocalTexel.x, min(materialLocalTexel.y + 1, materialCellSize - 1));
    const float localHeight = luminance(texel.rgb);
    const float neighborHeight = 0.5 * (
        luminance(texelFetch(blockAtlas, neighborX, atlasMip).rgb) +
        luminance(texelFetch(blockAtlas, neighborY, atlasMip).rgb));
    const float textureOcclusion = cloudMaterial
        ? 1.0
        : mix(0.88, 1.0, smoothstep(-0.18, 0.20, localHeight - neighborHeight));
    float skyFill = normal.y * 0.5 + 0.5;
    vec3 cameraOffset = inWorldPosition - frame.cameraPosition.xyz;
    float cameraDistance = length(cameraOffset);
    float horizontalDistance = length(cameraOffset.xz);
    bool underwater = frame.cameraPosition.w > 0.5;
    float distanceFog = underwater ? smoothstep(4.0, 28.0, cameraDistance)
                                   : smoothstep(300.0, 420.0, horizontalDistance);
    vec3 viewRay = normalize(cameraOffset);
    float skyGradient = clamp(viewRay.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 atmosphericSky = mix(frame.skyColor.rgb * 0.58, frame.skyColor.rgb, skyGradient);
    vec3 sky = underwater ? vec3(0.008, 0.055, 0.15) : atmosphericSky;
    vec3 material = texel.rgb;
    if (leavesMaterial) {
        material *= vec3(0.92, 1.0, 0.90);
    }
    if (cloudMaterial) {
        const vec3 nightCloud = vec3(0.20, 0.23, 0.30);
        const vec3 dayCloud = vec3(0.94, 0.96, 0.98);
        material = mix(nightCloud, dayCloud, frame.skyColor.a);
    } else if (waterMaterial) {
        const vec3 waterBase = vec3(0.006, 0.040, 0.24);
        material = mix(waterBase, texel.rgb * 0.58, texel.a < 0.1 ? 0.0 : 0.24);
    }
    float ambient = 0.10 + frame.lightColorIntensity.a * 0.16;
    const vec3 viewDirection = normalize(frame.cameraPosition.xyz - inWorldPosition);
    const vec3 halfDirection = normalize(toLight + viewDirection);
    const float roughness = waterMaterial ? 0.26 : (leavesMaterial ? 0.82 : 0.68);
    const float specularPower = mix(96.0, 12.0, roughness);
    const float normalView = max(dot(normal, viewDirection), 0.0);
    const vec3 baseReflectance = waterMaterial ? vec3(0.025) : vec3(0.04);
    const vec3 fresnel = baseReflectance +
                         (vec3(1.0) - baseReflectance) * pow(1.0 - normalView, 5.0);
    const float specular = pow(max(dot(normal, halfDirection), 0.0), specularPower) *
                           visibility * (1.0 - roughness * 0.72);
    vec3 lit = cloudMaterial
                   ? material
                   : material * (ambient + skyFill * 0.10 +
                                 diffuse * visibility * textureOcclusion *
                                     frame.lightColorIntensity.rgb *
                                     frame.lightColorIntensity.a) +
                         fresnel * specular * frame.lightColorIntensity.rgb *
                             frame.lightColorIntensity.a;
    if (underwater) {
        lit *= vec3(0.42, 0.62, 0.88);
    }
    const float materialFog = waterMaterial && !underwater
        ? max(distanceFog, smoothstep(220.0, 400.0, horizontalDistance))
        : distanceFog;
    vec3 linearColor = mix(lit, sky, materialFog);
    float outputAlpha = cloudMaterial ? 0.58 : (waterMaterial ? 0.94 : texel.a);
    outColor = vec4(linearColor, outputAlpha);
}
