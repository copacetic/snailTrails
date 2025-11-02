"""
GPU-Accelerated Snail Trails Simulation - Modular Version
Optimized for NVIDIA RTX 4090 - Supports millions of agents!

Refactored for testability with separated concerns.
"""

import moderngl_window as mglw
import numpy as np
import time
import os
import sys

# Import our modules
from src.config_manager import load_config_from_file
from src.simulation import AgentManager, OccupancyGrid, SimulationStats
from src.gpu_buffers import GPUBufferManager
from src.shaders import ShaderManager


class SnailTrailsGPU(mglw.WindowConfig):
    """GPU-accelerated agent simulation using compute shaders"""

    def __init__(self, **kwargs):
        # Load configuration first
        self.config = load_config_from_file('config.py')
        self.config.print_summary()

        # Set window configuration
        self.gl_version = (4, 3)
        self.title = f"Snail Trails GPU - {self.config['NUM_AGENTS']:,} Agents"
        self.window_size = (self.config['WINDOW_WIDTH'], self.config['WINDOW_HEIGHT'])
        self.aspect_ratio = self.config['WINDOW_WIDTH'] / self.config['WINDOW_HEIGHT']
        self.resizable = False
        self.vsync = self.config['VSYNC']
        self.fullscreen = self.config.get('FULLSCREEN', False)

        super().__init__(**kwargs)

        print("\n" + "="*70)
        print("  GPU-ACCELERATED SNAIL TRAILS")
        print("  Optimized for NVIDIA RTX 4090")
        print("="*70)
        print(f"GPU: {self.ctx.info['GL_RENDERER']}")
        print(f"OpenGL: {self.ctx.info['GL_VERSION']}")
        print("="*70 + "\n")

        # Initialize components
        self.stats = SimulationStats()
        self.last_fps_time = time.time()
        self.fps_frames = 0

        # Setup GPU resources
        self.setup_simulation()
        self.setup_gpu()
        self.setup_shaders()
        self.setup_rendering()

        # Generate initial vector field
        print("Generating initial vector field on GPU...")
        self.generate_vector_field()

        print("Initialization complete! Running simulation...\n")

    def setup_simulation(self):
        """Initialize simulation components"""
        print("Initializing simulation...")

        # Create agent manager
        self.agent_manager = AgentManager(
            num_agents=self.config['NUM_AGENTS'],
            grid_size=self.config['GRID_SIZE']
        )
        self.agent_manager.initialize_random_positions()

        # Create occupancy grid
        self.occupancy_grid = OccupancyGrid(self.config['GRID_SIZE'])
        self.occupancy_grid.mark_positions(self.agent_manager.get_positions())

        print(f"  Created {self.agent_manager.num_agents:,} agents")

    def setup_gpu(self):
        """Setup GPU buffers"""
        print("Setting up GPU buffers...")

        self.buffer_manager = GPUBufferManager(
            ctx=self.ctx,
            num_agents=self.config['NUM_AGENTS'],
            grid_size=self.config['GRID_SIZE']
        )

        # Upload initial data
        self.buffer_manager.upload_agents(self.agent_manager.get_bytes())
        self.buffer_manager.upload_occupancy(self.occupancy_grid.get_bytes())

        memory_mb = self.buffer_manager.get_memory_usage_mb()
        print(f"  Allocated ~{memory_mb:.1f} MB GPU memory")

    def setup_shaders(self):
        """Load and compile shaders"""
        print("Compiling shaders...")

        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        self.shader_manager = ShaderManager(ctx=self.ctx, shader_dir=shader_dir)

        # Compile compute shaders
        self.field_compute = self.shader_manager.compile_compute_shader(
            'field_compute.glsl',
            name='field_compute'
        )

        self.agent_compute = self.shader_manager.compile_compute_shader(
            'agent_compute.glsl',
            name='agent_compute'
        )

        # Compile render program
        self.render_program = self.shader_manager.compile_render_program(
            'vertex.glsl',
            'fragment.glsl',
            name='render'
        )

        print("  Shaders compiled successfully")

    def setup_rendering(self):
        """Setup rendering VAO"""
        self.vao = self.buffer_manager.create_vao(self.render_program)

    def generate_vector_field(self):
        """Generate vector field on GPU using compute shader"""
        self.buffer_manager.bind_for_field_compute()

        self.field_compute['gridSize'] = self.config['GRID_SIZE']
        self.field_compute['time'] = self.stats.field_generation_count * 10.0
        self.field_compute['samples'] = self.config['FIELD_SAMPLES']

        # Dispatch compute shader
        groups_x = (self.config['GRID_SIZE'] + 15) // 16
        groups_y = (self.config['GRID_SIZE'] + 15) // 16
        self.field_compute.run(groups_x, groups_y)

        self.stats.field_regenerated()

    def update_agents(self):
        """Update all agents on GPU using compute shader"""
        # Reset stats
        self.buffer_manager.reset_stats()

        # Bind buffers
        self.buffer_manager.bind_for_agent_compute()

        # Set uniforms
        self.agent_compute['gridSize'] = self.config['GRID_SIZE']
        self.agent_compute['numAgents'] = self.config['NUM_AGENTS']

        # Dispatch compute shader
        groups = (self.config['NUM_AGENTS'] + 255) // 256
        self.agent_compute.run(groups)

        # Read back stats
        stats_data = self.buffer_manager.read_stats()
        not_moved = int(stats_data[0])
        moving_agents = self.config['NUM_AGENTS'] - not_moved

        # Update statistics
        self.stats.update(agents_moved=moving_agents, agents_stuck=not_moved)

        # Regenerate field if agents are stuck
        if moving_agents < self.config['STUCK_THRESHOLD']:
            print(f"Frame {self.stats.frame_count}: Regenerating field "
                  f"(only {moving_agents:,} agents moving)")
            self.generate_vector_field()

        # Periodic status update
        if self.stats.frame_count % 60 == 0:
            percent_moving = 100 * moving_agents / self.config['NUM_AGENTS']
            print(f"Frame {self.stats.frame_count}: "
                  f"{moving_agents:,} / {self.config['NUM_AGENTS']:,} agents moving "
                  f"({percent_moving:.1f}%)")

        return not_moved

    def render(self, time_elapsed, frame_time):
        """Render frame"""
        self.ctx.clear(1.0, 1.0, 1.0)

        # Update simulation
        self.update_agents()

        # Set up projection matrix
        projection = np.array([
            [2.0/self.config['WINDOW_WIDTH'], 0, 0, 0],
            [0, 2.0/self.config['WINDOW_HEIGHT'], 0, 0],
            [0, 0, -1, 0],
            [-1, -1, 0, 1]
        ], dtype='f4')

        self.render_program['projection'].write(projection.tobytes())
        self.render_program['gridSize'] = self.config['GRID_SIZE']
        self.render_program['windowSize'] = (
            self.config['WINDOW_WIDTH'],
            self.config['WINDOW_HEIGHT']
        )
        self.render_program['agentSize'] = self.config['AGENT_SIZE']

        # Draw all agents with instancing
        self.vao.render(instances=self.config['NUM_AGENTS'])

        # Update FPS display
        self.fps_frames += 1
        if self.config['SHOW_FPS'] and time_elapsed - self.last_fps_time >= 1.0:
            fps = self.fps_frames / (time_elapsed - self.last_fps_time)
            self.wnd.title = (
                f"Snail Trails GPU - {self.config['NUM_AGENTS']:,} Agents | "
                f"FPS: {fps:.1f}"
            )
            self.last_fps_time = time_elapsed
            self.fps_frames = 0

    def close(self):
        """Cleanup on close"""
        print("\nShutting down...")
        summary = self.stats.get_summary()
        print(f"  Total frames: {summary['frames']}")
        print(f"  Field regenerations: {summary['field_generations']}")
        print(f"  Avg agents moved per frame: {summary['avg_moved_per_frame']:,.0f}")

        self.buffer_manager.release()
        super().close()


if __name__ == '__main__':
    # Run the simulation
    SnailTrailsGPU.run()
