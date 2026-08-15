#version 450

layout(location = 0) out vec2 outClipPosition;

void main() {
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    outClipPosition = positions[gl_VertexIndex];
    gl_Position = vec4(outClipPosition, 0.9999, 1.0);
}
