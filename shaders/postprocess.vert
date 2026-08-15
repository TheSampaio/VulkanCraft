#version 450

layout(location = 0) out vec2 outTextureCoordinate;

void main() {
    const vec2 position = vec2(
        float((gl_VertexIndex << 1) & 2),
        float(gl_VertexIndex & 2));
    outTextureCoordinate = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
