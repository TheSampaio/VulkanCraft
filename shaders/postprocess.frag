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
    uvec4 consoleText[16];
    uvec4 consoleState;
} frame;

layout(set = 0, binding = 3) uniform sampler2D sceneColor;
layout(set = 0, binding = 4) uniform sampler2D sceneDepth;
layout(set = 0, binding = 1) uniform sampler2DArray shadowMap;

layout(location = 0) in vec2 inTextureCoordinate;
layout(location = 0) out vec4 outColor;

const float nearPlane = 0.1;
const float farPlane = 832.0;

float linearDepth(float depth) {
    return nearPlane * farPlane / max(farPlane - depth * (farPlane - nearPlane), 0.0001);
}

vec3 reconstructViewPosition(vec2 uv, float depth) {
    vec4 viewPosition = inverse(frame.projection) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return viewPosition.xyz / max(viewPosition.w, 0.0001);
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

float atmosphericFogFactor(vec3 viewPosition, vec3 worldOffset, bool waterMaterial) {
    if (frame.cameraPosition.w > 0.5) {
        return smoothstep(4.0, 28.0, length(viewPosition));
    }
    const float horizontalDistance = length(worldOffset.xz);
    return waterMaterial ? smoothstep(220.0, 400.0, horizontalDistance)
                         : smoothstep(300.0, 420.0, horizontalDistance);
}

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

uint glyphBits(uint character) {
    const uint letters[26] = uint[26](
        0x99F96u, 0x79797u, 0xE111Eu, 0x79997u, 0xF171Fu, 0x1171Fu,
        0xE9D1Eu, 0x99F99u, 0xE444Eu, 0x6988Cu, 0x95359u, 0xF1111u,
        0x99FF9u, 0x99DB9u, 0x69996u, 0x11797u, 0xED996u, 0x95797u,
        0x7861Eu, 0x6666Fu, 0x69999u, 0x66999u, 0x9FF99u, 0x96669u,
        0x66699u, 0xF124Fu);
    if (character >= 65u && character <= 90u) character += 32u;
    if (character >= 97u && character <= 122u) return letters[character - 97u];
    if (character == 47u) return 0x1248u;
    if (character == 58u) return 0x6060u;
    if (character == 62u) return 0x12421u;
    return 0u;
}

vec3 drawConsole(vec2 uv, vec3 color) {
    if (frame.consoleState.x == 0u || uv.y >= 0.22) return color;
    vec3 result = mix(color, vec3(0.012, 0.016, 0.024), 0.88);
    const vec2 characterPosition = (uv - vec2(0.025, 0.065)) / vec2(0.0145, 0.052);
    const int characterIndex = int(floor(characterPosition.x));
    if (characterPosition.y < 0.0 || characterPosition.y >= 1.0 ||
        characterIndex < 0 || characterIndex >= int(frame.consoleState.y)) {
        return result;
    }
    const uvec4 packed = frame.consoleText[characterIndex / 4];
    const uint character = packed[characterIndex % 4];
    const ivec2 glyph = ivec2(floor(vec2(fract(characterPosition.x) * 5.0,
                                              characterPosition.y * 7.0)));
    if (glyph.x >= 4 || glyph.y >= 6) return result;
    const uint bit = uint(glyph.y * 4 + glyph.x);
    if (((glyphBits(character) >> bit) & 1u) != 0u) {
        result = vec3(0.88, 0.94, 1.0);
    }
    return result;
}

float ambientOcclusion(vec2 uv, float centerDepth, vec2 texelSize) {
    if (centerDepth >= 0.9999) return 1.0;
    const vec2 directions[8] = vec2[8](
        vec2(1.0, 0.0), vec2(0.707, 0.707), vec2(0.0, 1.0), vec2(-0.707, 0.707),
        vec2(-1.0, 0.0), vec2(-0.707, -0.707), vec2(0.0, -1.0), vec2(0.707, -0.707));
    const vec3 centerPosition = reconstructViewPosition(uv, centerDepth);
    vec3 normal = normalize(cross(dFdx(centerPosition), dFdy(centerPosition)));
    if (dot(normal, -centerPosition) < 0.0) normal = -normal;
    float occlusion = 0.0;
    for (int index = 0; index < 8; ++index) {
        const float radius = index % 2 == 0 ? 5.0 : 10.0;
        const vec2 sampleUv = clamp(
            uv + directions[index] * texelSize * radius,
            texelSize,
            vec2(1.0) - texelSize);
        const float sampleDepth = texture(sceneDepth, sampleUv).r;
        if (sampleDepth >= 0.9999) continue;
        const vec3 difference = reconstructViewPosition(sampleUv, sampleDepth) - centerPosition;
        const float distanceToSample = length(difference);
        const float horizon = max(dot(normal, difference / max(distanceToSample, 0.0001)) - 0.08, 0.0);
        occlusion += horizon * (1.0 - smoothstep(1.0, 18.0, distanceToSample));
    }
    return clamp(1.0 - occlusion * 0.16, 0.80, 1.0);
}

vec3 localBloom(vec2 uv, vec2 texelSize) {
    const vec2 offsets[4] = vec2[4](
        vec2(3.0, 3.0), vec2(-3.0, 3.0), vec2(3.0, -3.0), vec2(-3.0, -3.0));
    vec3 bloom = vec3(0.0);
    for (int index = 0; index < 4; ++index) {
        const vec3 sampleColor = texture(sceneColor, uv + offsets[index] * texelSize).rgb;
        bloom += sampleColor * smoothstep(0.90, 1.45, luminance(sampleColor));
    }
    return bloom * 0.018;
}

vec4 screenSpaceWaterReflection(
    vec2 uv,
    vec2 texelSize,
    float waterMask,
    vec3 viewPosition,
    vec3 viewNormal) {
    if (waterMask <= 0.0) return vec4(0.0);
    const vec3 reflectedDirection = normalize(reflect(normalize(viewPosition), viewNormal));
    for (int stepIndex = 1; stepIndex <= 28; ++stepIndex) {
        const float distanceAlongRay = 0.8 + float(stepIndex) * 2.6;
        const vec3 rayPosition = viewPosition + reflectedDirection * distanceAlongRay;
        const vec4 rayClip = frame.projection * vec4(rayPosition, 1.0);
        if (rayClip.w <= 0.0) break;
        const vec2 sampleUv = rayClip.xy / rayClip.w * 0.5 + 0.5;
        if (any(lessThan(sampleUv, texelSize)) ||
            any(greaterThan(sampleUv, vec2(1.0) - texelSize))) break;
        const float sampleDepth = texture(sceneDepth, sampleUv).r;
        if (sampleDepth >= 0.9999) continue;
        const vec3 sampledPosition = reconstructViewPosition(sampleUv, sampleDepth);
        const float depthGap = sampledPosition.z - rayPosition.z;
        const float sampledAlpha = texture(sceneColor, sampleUv).a;
        const float hitThickness = 1.2 + distanceAlongRay * 0.028;
        if (depthGap >= 0.0 && depthGap < hitThickness && abs(sampledAlpha - 0.94) > 0.03) {
            const float edgeFade = smoothstep(0.0, 0.08, sampleUv.x) *
                                   smoothstep(0.0, 0.08, sampleUv.y) *
                                   smoothstep(0.0, 0.08, 1.0 - sampleUv.x) *
                                   smoothstep(0.0, 0.08, 1.0 - sampleUv.y);
            const float distanceFade = 1.0 - float(stepIndex) / 29.0;
            const float hitConfidence = 1.0 - depthGap / hitThickness;
            const vec2 roughnessOffset = texelSize *
                (1.5 + distanceAlongRay * 0.015);
            vec3 reflectedColor = texture(sceneColor, sampleUv).rgb * 0.40;
            reflectedColor += texture(sceneColor, sampleUv + vec2(roughnessOffset.x, 0.0)).rgb * 0.15;
            reflectedColor += texture(sceneColor, sampleUv - vec2(roughnessOffset.x, 0.0)).rgb * 0.15;
            reflectedColor += texture(sceneColor, sampleUv + vec2(0.0, roughnessOffset.y)).rgb * 0.15;
            reflectedColor += texture(sceneColor, sampleUv - vec2(0.0, roughnessOffset.y)).rgb * 0.15;
            return vec4(
                reflectedColor,
                edgeFade * distanceFade * hitConfidence);
        }
    }
    return vec4(0.0);
}

float shadowVisibility(vec3 worldPosition, float viewDistance) {
    if (viewDistance > frame.cascadeSplits.w) return 1.0;
    int cascade = 0;
    if (viewDistance > frame.cascadeSplits.x) cascade = 1;
    if (viewDistance > frame.cascadeSplits.y) cascade = 2;
    if (viewDistance > frame.cascadeSplits.z) cascade = 3;
    const vec4 lightClip = frame.lightViewProjection[cascade] * vec4(worldPosition, 1.0);
    const vec3 projected = lightClip.xyz / max(lightClip.w, 0.0001);
    const vec2 shadowUv = projected.xy * 0.5 + 0.5;
    if (projected.z <= 0.0 || projected.z >= 1.0 ||
        any(lessThan(shadowUv, vec2(0.0))) || any(greaterThan(shadowUv, vec2(1.0)))) {
        return 1.0;
    }
    const float storedDepth = texture(shadowMap, vec3(shadowUv, float(cascade))).r;
    return projected.z - 0.0015 <= storedDepth ? 1.0 : 0.0;
}

float interleavedNoise(vec2 pixel) {
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

float volumetricSunlight(vec2 uv, float depth) {
    if (frame.cameraPosition.w > 0.5 || frame.skyColor.a <= 0.001) return 0.0;
    const vec3 viewPosition = reconstructViewPosition(uv, min(depth, 0.99999));
    const float rayLength = min(length(viewPosition), 220.0);
    if (rayLength <= 0.5) return 0.0;
    const vec3 viewDirection = normalize(
        (inverse(frame.view) * vec4(normalize(viewPosition), 0.0)).xyz);
    const float jitter = interleavedNoise(uv * vec2(textureSize(sceneDepth, 0)));
    float illuminatedDensity = 0.0;
    const int sampleCount = 6;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        const float progress =
            (float(sampleIndex) + 0.35 + jitter * 0.30) / float(sampleCount);
        const float sampleDistance = rayLength * progress;
        const vec3 samplePosition =
            frame.cameraPosition.xyz + viewDirection * sampleDistance;
        const float heightDensity = exp(-max(samplePosition.y - 72.0, 0.0) * 0.006);
        illuminatedDensity +=
            shadowVisibility(samplePosition, sampleDistance) * heightDensity;
    }
    const float averageVisibility = illuminatedDensity / float(sampleCount);
    const float forwardScattering =
        pow(max(dot(viewDirection, normalize(frame.sunDirection.xyz)), 0.0), 6.0);
    const float opticalDepth = 1.0 - exp(-rayLength * 0.006);
    return averageVisibility * opticalDepth *
           (0.035 + forwardScattering * 0.30) * frame.skyColor.a;
}

float godRays(vec2 uv, vec2 sunUv, vec2 texelSize) {
    if (any(lessThan(sunUv, vec2(0.0))) || any(greaterThan(sunUv, vec2(1.0)))) return 0.0;
    const vec2 direction = (sunUv - uv) / 10.0;
    float illumination = 0.0;
    float decay = 1.0;
    vec2 sampleUv = uv;
    for (int index = 0; index < 10; ++index) {
        sampleUv = clamp(sampleUv + direction, texelSize, vec2(1.0) - texelSize);
        const float sky = step(0.9995, texture(sceneDepth, sampleUv).r);
        const float brightness = smoothstep(0.42, 1.1, luminance(texture(sceneColor, sampleUv).rgb));
        illumination += sky * (0.35 + brightness * 0.65) * decay;
        decay *= 0.94;
    }
    const float radialFalloff = 1.0 - smoothstep(0.05, 0.82, distance(uv, sunUv));
    return illumination * 0.018 * radialFalloff * frame.skyColor.a;
}

void main() {
    const vec2 textureSizeValue = vec2(textureSize(sceneColor, 0));
    const vec2 texelSize = 1.0 / textureSizeValue;
    const vec4 source = texture(sceneColor, inTextureCoordinate);
    const float depth = texture(sceneDepth, inTextureCoordinate).r;
    const float viewDepth = linearDepth(depth);
    const float waterMask = 1.0 - smoothstep(0.015, 0.035, abs(source.a - 0.94));
    const vec3 viewPosition = reconstructViewPosition(
        inTextureCoordinate, min(depth, 0.99999));
    const vec3 worldOffset = (inverse(frame.view) * vec4(viewPosition, 0.0)).xyz;
    const vec3 worldPosition = frame.cameraPosition.xyz + worldOffset;
    const float terrainFog = depth < 0.9999
        ? atmosphericFogFactor(viewPosition, worldOffset, false)
        : 1.0;
    const float waterFog = depth < 0.9999
        ? atmosphericFogFactor(viewPosition, worldOffset, true)
        : 1.0;
    const float objectFog = mix(terrainFog, waterFog, waterMask);
    const float ao = objectFog >= 0.995
        ? 1.0
        : ambientOcclusion(inTextureCoordinate, depth, texelSize);
    vec3 color = source.rgb * mix(ao, 1.0, objectFog);

    const vec3 geometricViewCross = cross(dFdx(viewPosition), dFdy(viewPosition));
    const float geometricLength = length(geometricViewCross);
    const vec3 geometricViewNormal = geometricLength > 0.0001
        ? geometricViewCross / geometricLength
        : vec3(0.0, 0.0, 1.0);
    const vec3 geometricWorldNormal = normalize(
        mat3(inverse(frame.view)) * geometricViewNormal);
    const float topSurface = depth < 0.9999
        ? smoothstep(0.72, 0.94, abs(geometricWorldNormal.y))
        : 0.0;
    const float visibleWaterMask = waterMask * (1.0 - waterFog);
    const float ssrWaterMask = visibleWaterMask * topSurface;
    const vec3 waveNormal = waterWaveNormal(worldPosition);
    const vec3 waveViewNormal = normalize(mat3(frame.view) * waveNormal);
    const vec4 reflection = screenSpaceWaterReflection(
        inTextureCoordinate,
        texelSize,
        ssrWaterMask,
        viewPosition,
        waveViewNormal);
    const vec3 toCamera = normalize(-worldOffset);
    const float normalView = max(dot(waveNormal, toCamera), 0.0);
    const float fresnel = 0.02 + 0.98 * pow(1.0 - normalView, 5.0);
    const float reflectionWeight = ssrWaterMask * reflection.a * min(fresnel * 0.55, 0.22);
    color = mix(color, reflection.rgb, reflectionWeight);
    const float deepWater = visibleWaterMask * smoothstep(8.0, 70.0, viewDepth);
    color = mix(color, vec3(0.006, 0.045, 0.16), deepWater * 0.62);

    const vec4 sunClip = frame.projection * frame.view *
                         vec4(frame.cameraPosition.xyz + frame.sunDirection.xyz * 500.0, 1.0);
    const vec2 sunUv = sunClip.xy / max(sunClip.w, 0.0001) * 0.5 + 0.5;
    const float rays = sunClip.w > 0.0 && frame.cameraPosition.w < 0.5
        ? godRays(inTextureCoordinate, sunUv, texelSize)
        : 0.0;
    const float volumetric = volumetricSunlight(inTextureCoordinate, depth);
    color += frame.lightColorIntensity.rgb *
             frame.lightColorIntensity.a * (rays + volumetric);

    const float aerialPerspective = depth < 0.9999 ? objectFog * 0.055 : 0.0;
    color = mix(color, frame.skyColor.rgb, aerialPerspective);
    color += localBloom(inTextureCoordinate, texelSize);

    const vec3 exposed = color * 0.96;
    const vec3 toneMapped = clamp(
        (exposed * (2.51 * exposed + 0.03)) /
        (exposed * (2.43 * exposed + 0.59) + 0.14),
        vec3(0.0),
        vec3(1.0));
    const vec3 displayColor = pow(toneMapped, vec3(1.0 / 2.2));
    outColor = vec4(drawConsole(inTextureCoordinate, displayColor), 1.0);
}
