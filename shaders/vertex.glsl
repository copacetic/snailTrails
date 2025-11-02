#version 430

in vec2 in_position;  // Instance data: agent position
in vec2 in_velocity;  // Instance data: agent velocity
in float in_active;   // Instance data: is active

in vec2 in_vertex;    // Per-vertex data: square corners

uniform mat4 projection;
uniform int gridSize;
uniform vec2 windowSize;
uniform float agentSize;

out vec3 v_color;
out vec2 v_velocity;

void main() {
    if(in_active < 0.5) {
        gl_Position = vec4(-10, -10, 0, 1);  // Cull inactive
        return;
    }

    // Scale from grid coordinates to screen coordinates
    vec2 scale = windowSize / float(gridSize);
    vec2 screenPos = in_position * scale;

    // Create small square for each agent
    vec2 finalPos = screenPos + in_vertex * scale * agentSize;

    gl_Position = projection * vec4(finalPos, 0.0, 1.0);

    // Color based on velocity direction
    float angle = atan(in_velocity.y, in_velocity.x);
    v_color = vec3(
        0.5 + 0.5 * cos(angle),
        0.5 + 0.5 * cos(angle + 2.094),
        0.5 + 0.5 * cos(angle + 4.189)
    );
    v_velocity = in_velocity;
}
