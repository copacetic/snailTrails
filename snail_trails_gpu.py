"""
GPU-Accelerated Snail Trails Simulation
Optimized for NVIDIA RTX 4090 - Supports millions of agents!

Uses compute shaders for parallel processing on GPU
"""

import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import numpy as np
import random
import math
import time

# Import configuration
try:
    from config import *
except ImportError:
    print("Warning: config.py not found, using defaults")
    # Defaults if config.py doesn't exist
    GRID_SIZE = 2048
    NUM_AGENTS = 10_000_000
    WINDOW_WIDTH = 1920
    WINDOW_HEIGHT = 1080
    VSYNC = True
    SHOW_FPS = True
    STUCK_THRESHOLD_PERCENT = 1.0
    FIELD_SAMPLES = 500
    AGENT_SIZE = 0.8
    COLOR_MODE = 'velocity'

# Vector directions (8-directional movement)
POSSIBLE_VECTORS = [
    (1, 1), (1, 0), (0, 1), (-1, 0),
    (0, -1), (-1, 1), (1, -1), (-1, -1)
]


class SnailTrailsGPU(mglw.WindowConfig):
    """GPU-accelerated agent simulation using compute shaders"""

    gl_version = (4, 3)  # Need 4.3+ for compute shaders
    title = f"Snail Trails GPU - {NUM_AGENTS:,} Agents on RTX 4090"
    window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
    aspect_ratio = WINDOW_WIDTH / WINDOW_HEIGHT
    resizable = False
    vsync = VSYNC

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("\n" + "="*70)
        print("  GPU-ACCELERATED SNAIL TRAILS")
        print("  Optimized for NVIDIA RTX 4090")
        print("="*70)
        print(f"Grid: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE*GRID_SIZE:,} cells)")
        print(f"Agents: {NUM_AGENTS:,}")
        print(f"GPU: {self.ctx.info['GL_RENDERER']}")
        print(f"OpenGL: {self.ctx.info['GL_VERSION']}")
        print("="*70 + "\n")

        self.frame_count = 0
        self.field_generation_count = 0
        self.field_threshold = int(NUM_AGENTS * (STUCK_THRESHOLD_PERCENT / 100.0))

        # FPS tracking
        self.last_fps_time = time.time()
        self.fps_frames = 0

        # Initialize GPU resources
        self.setup_compute_shaders()
        self.setup_render_shaders()
        self.setup_buffers()
        self.initialize_agents()

        # Generate initial vector field
        print("Generating initial vector field on GPU...")
        self.generate_vector_field()

        print("Initialization complete! Running simulation...")

    def setup_compute_shaders(self):
        """Compile compute shaders for vector field and agent movement"""

        # Vector field generation shader
        self.field_compute = self.ctx.compute_shader('''
            #version 430

            layout(local_size_x = 16, local_size_y = 16) in;

            layout(std430, binding = 0) buffer VectorField {
                vec2 vectors[];
            };

            uniform int gridSize;
            uniform float time;

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
                int samples = """ + str(FIELD_SAMPLES) + """;
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
        ''')

        # Agent movement shader
        self.agent_compute = self.ctx.compute_shader('''
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
        ''')

    def setup_render_shaders(self):
        """Compile vertex/fragment shaders for instanced rendering"""

        self.render_program = self.ctx.program(
            vertex_shader='''
                #version 430

                in vec2 in_position;  // Instance data: agent position
                in vec2 in_velocity;  // Instance data: agent velocity
                in float in_active;   // Instance data: is active

                in vec2 in_vertex;    // Per-vertex data: square corners

                uniform mat4 projection;
                uniform int gridSize;
                uniform vec2 windowSize;

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
                    vec2 finalPos = screenPos + in_vertex * scale * """ + str(AGENT_SIZE) + """;

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
            ''',
            fragment_shader='''
                #version 430

                in vec3 v_color;
                in vec2 v_velocity;

                out vec4 fragColor;

                void main() {
                    float speed = length(v_velocity);
                    fragColor = vec4(v_color * (0.5 + 0.5 * speed), 1.0);
                }
            ''')

    def setup_buffers(self):
        """Create GPU buffers for agents, field, and occupancy grid"""

        # Agent buffer (position, velocity, active flag)
        # Using struct padding for alignment
        agent_dtype = np.dtype([
            ('pos', np.float32, 2),
            ('velocity', np.float32, 2),
            ('active', np.float32),
            ('padding', np.float32)
        ])

        self.agents_data = np.zeros(NUM_AGENTS, dtype=agent_dtype)
        self.agents_buffer = self.ctx.buffer(self.agents_data.tobytes())

        # Vector field buffer
        field_size = GRID_SIZE * GRID_SIZE * 2 * 4  # vec2, float32
        self.field_buffer = self.ctx.buffer(reserve=field_size)

        # Occupancy grid buffer (int32)
        grid_size = GRID_SIZE * GRID_SIZE * 4
        self.occupancy_buffer = self.ctx.buffer(reserve=grid_size)

        # Stats buffer (notMoved count, total agents)
        self.stats_buffer = self.ctx.buffer(reserve=8)  # 2 ints

        # Vertex data for instanced rendering (square corners)
        vertices = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5]
        ], dtype='f4')

        self.square_vbo = self.ctx.buffer(vertices.tobytes())

        # Create VAO for instanced rendering
        self.vao = self.ctx.vertex_array(
            self.render_program,
            [
                (self.agents_buffer, '2f 2f 1f 1f/i', 'in_position', 'in_velocity', 'in_active'),
                (self.square_vbo, '2f', 'in_vertex')
            ]
        )

    def initialize_agents(self):
        """Initialize agent positions randomly on CPU, upload to GPU"""

        print("Initializing agents...")

        # Random positions
        positions = np.random.randint(0, GRID_SIZE, size=(NUM_AGENTS, 2), dtype=np.float32)
        self.agents_data['pos'] = positions
        self.agents_data['velocity'] = 0
        self.agents_data['active'] = 1.0

        # Upload to GPU
        self.agents_buffer.write(self.agents_data.tobytes())

        # Initialize occupancy grid
        occupancy = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.int32)
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                occupancy[y * GRID_SIZE + x] = 1

        self.occupancy_buffer.write(occupancy.tobytes())

        print(f"Initialized {NUM_AGENTS:,} agents")

    def generate_vector_field(self):
        """Generate vector field on GPU using compute shader"""

        self.field_buffer.bind_to_storage_buffer(0)

        self.field_compute['gridSize'] = GRID_SIZE
        self.field_compute['time'] = self.field_generation_count * 10.0

        # Dispatch compute shader (16x16 work groups)
        groups_x = (GRID_SIZE + 15) // 16
        groups_y = (GRID_SIZE + 15) // 16
        self.field_compute.run(groups_x, groups_y)

        self.field_generation_count += 1

    def update_agents(self):
        """Update all agents on GPU using compute shader"""

        # Reset stats
        stats = np.array([0, NUM_AGENTS], dtype=np.int32)
        self.stats_buffer.write(stats.tobytes())

        # Bind buffers
        self.agents_buffer.bind_to_storage_buffer(0)
        self.field_buffer.bind_to_storage_buffer(1)
        self.occupancy_buffer.bind_to_storage_buffer(2)
        self.stats_buffer.bind_to_storage_buffer(3)

        self.agent_compute['gridSize'] = GRID_SIZE
        self.agent_compute['numAgents'] = NUM_AGENTS

        # Dispatch compute shader (256 threads per group)
        groups = (NUM_AGENTS + 255) // 256
        self.agent_compute.run(groups)

        # Read back stats
        stats_data = np.frombuffer(self.stats_buffer.read(), dtype=np.int32)
        not_moved = stats_data[0]

        # Regenerate field if agents are stuck
        moving_agents = NUM_AGENTS - not_moved
        if moving_agents < self.field_threshold:
            print(f"Frame {self.frame_count}: Regenerating field (only {moving_agents:,} agents moving)")
            self.generate_vector_field()

        if self.frame_count % 60 == 0:
            print(f"Frame {self.frame_count}: {moving_agents:,} / {NUM_AGENTS:,} agents moving ({100*moving_agents/NUM_AGENTS:.1f}%)")

        return not_moved

    def render(self, time, frame_time):
        """Render frame"""

        self.ctx.clear(1.0, 1.0, 1.0)

        # Update simulation
        self.update_agents()

        # Set up projection matrix
        projection = np.array([
            [2.0/WINDOW_WIDTH, 0, 0, 0],
            [0, 2.0/WINDOW_HEIGHT, 0, 0],
            [0, 0, -1, 0],
            [-1, -1, 0, 1]
        ], dtype='f4')

        self.render_program['projection'].write(projection.tobytes())
        self.render_program['gridSize'] = GRID_SIZE
        self.render_program['windowSize'] = (WINDOW_WIDTH, WINDOW_HEIGHT)

        # Draw all agents with instancing (single draw call!)
        self.vao.render(instances=NUM_AGENTS)

        self.frame_count += 1
        self.fps_frames += 1

        # Update FPS display
        if SHOW_FPS and time - self.last_fps_time >= 1.0:
            fps = self.fps_frames / (time - self.last_fps_time)
            self.wnd.title = f"Snail Trails GPU - {NUM_AGENTS:,} Agents | FPS: {fps:.1f}"
            self.last_fps_time = time
            self.fps_frames = 0


if __name__ == '__main__':
    # Run the simulation
    SnailTrailsGPU.run()
