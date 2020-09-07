#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (push_constant) uniform PushConsts {
  mat4 mvp;
} push;

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 0) out vec2 v_uv;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    v_uv = a_uv;
    gl_Position = push.mvp * vec4(a_pos, 1.0);
}
