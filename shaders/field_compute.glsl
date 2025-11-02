#version 430

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer VectorField {
    vec2 vectors[];
};

uniform int gridSize;
uniform float time;
uniform int samples;

void main() {
    uvec2 id = gl_GlobalInvocationID.xy;
    if(id.x >= gridSize || id.y >= gridSize) return;

    int idx = int(id.y * gridSize + id.x);

    // Initialize with random direction
    float angle = fract(sin(dot(vec2(id.xy), vec2(12.9898, 78.233))) * 43758.5453) * 6.28318;
    vec2 randomDir = normalize(vec2(cos(angle), sin(angle)));
    vectors[idx] = randomDir;

    // Generate parametric curve-based vector field
    float cellX = float(id.x);
    float cellY = float(id.y);

    float minDist = 10000.0;
    vec2 bestDir = randomDir;

    // Sample parametric curve
    for(int t = 0; t < samples; t++) {
        float tNorm = float(t) / 2.0 + time;
        float rad = radians(tNorm);

        float a = 10.0;
        float b = 0.1;
        float coeff = a * exp(b * rad);

        float px = 200.0 * cos(3.0 * rad) + float(gridSize) / 2.0;
        float py = 300.0 * sin(5.0 * rad) + float(gridSize) / 2.0;

        float dist = distance(vec2(cellX, cellY), vec2(px, py));

        if(dist < minDist && dist < 5.0) {
            minDist = dist;

            // Calculate tangent direction
            float nextRad = radians(tNorm + 0.5);
            float nextPx = 200.0 * cos(3.0 * nextRad) + float(gridSize) / 2.0;
            float nextPy = 300.0 * sin(5.0 * nextRad) + float(gridSize) / 2.0;

            vec2 tangent = vec2(nextPx - px, nextPy - py);
            if(length(tangent) > 0.001) {
                bestDir = normalize(tangent);
            }
        }
    }

    // Quantize to 8 directions
    float bestAngle = atan(bestDir.y, bestDir.x);
    int dirIdx = int(round(bestAngle / (3.14159 / 4.0))) % 8;

    // Map to discrete directions
    vec2 dirs[8] = vec2[8](
        vec2(1, 0), vec2(1, 1), vec2(0, 1), vec2(-1, 1),
        vec2(-1, 0), vec2(-1, -1), vec2(0, -1), vec2(1, -1)
    );

    vectors[idx] = normalize(dirs[(dirIdx + 8) % 8]);
}
