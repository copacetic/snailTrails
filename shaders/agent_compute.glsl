#version 430

layout(local_size_x = 256) in;

struct Agent {
    vec2 pos;
    vec2 velocity;
    float active;
    float padding;
};

layout(std430, binding = 0) buffer Agents {
    Agent agents[];
};

layout(std430, binding = 1) buffer VectorField {
    vec2 vectors[];
};

layout(std430, binding = 2) buffer OccupancyGrid {
    int occupied[];
};

layout(std430, binding = 3) buffer Stats {
    int notMoved;
    int totalAgents;
};

uniform int gridSize;
uniform int numAgents;

bool inBounds(ivec2 pos) {
    return pos.x >= 0 && pos.x < gridSize && pos.y >= 0 && pos.y < gridSize;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if(id >= numAgents || agents[id].active < 0.5) return;

    ivec2 gridPos = ivec2(agents[id].pos);
    if(!inBounds(gridPos)) {
        agents[id].active = 0.0;
        return;
    }

    int gridIdx = gridPos.y * gridSize + gridPos.x;

    // Get vector field direction
    vec2 direction = vectors[gridIdx];
    ivec2 moveDir = ivec2(round(direction));
    ivec2 newGridPos = gridPos + moveDir;

    // Check bounds
    if(!inBounds(newGridPos)) {
        atomicAdd(stats[0].notMoved, 1);
        return;
    }

    int newGridIdx = newGridPos.y * gridSize + newGridPos.x;

    // Try to move using atomic compare-and-swap
    int oldVal = atomicCompSwap(occupied[newGridIdx], 0, 1);

    if(oldVal == 0) {
        // Successfully claimed new position
        atomicExchange(occupied[gridIdx], 0);
        agents[id].pos = vec2(newGridPos);
        agents[id].velocity = direction;
    } else {
        // Position occupied, couldn't move
        atomicAdd(stats[0].notMoved, 1);
    }
}
