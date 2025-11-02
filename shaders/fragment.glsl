#version 430

in vec3 v_color;
in vec2 v_velocity;

out vec4 fragColor;

void main() {
    float speed = length(v_velocity);
    fragColor = vec4(v_color * (0.5 + 0.5 * speed), 1.0);
}
