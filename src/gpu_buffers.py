"""
GPU buffer management for OpenGL compute shaders
"""

import numpy as np
from typing import Optional
import moderngl


class GPUBufferManager:
    """Manages GPU buffers for simulation data"""

    def __init__(self, ctx: moderngl.Context, num_agents: int, grid_size: int):
        """
        Initialize GPU buffer manager

        Args:
            ctx: ModernGL context
            num_agents: Number of agents
            grid_size: Grid size
        """
        self.ctx = ctx
        self.num_agents = num_agents
        self.grid_size = grid_size

        # Create buffers
        self.agents_buffer = None
        self.field_buffer = None
        self.occupancy_buffer = None
        self.stats_buffer = None
        self.square_vbo = None

        self._create_buffers()

    def _create_buffers(self) -> None:
        """Create all GPU buffers"""
        # Agent buffer (position, velocity, active, padding) * num_agents
        # 24 bytes per agent (2+2+1+1 floats)
        agent_size = self.num_agents * 24
        self.agents_buffer = self.ctx.buffer(reserve=agent_size)

        # Vector field buffer (vec2 per cell)
        field_size = self.grid_size * self.grid_size * 2 * 4  # 2 floats per cell
        self.field_buffer = self.ctx.buffer(reserve=field_size)

        # Occupancy grid buffer (int32 per cell)
        grid_size = self.grid_size * self.grid_size * 4
        self.occupancy_buffer = self.ctx.buffer(reserve=grid_size)

        # Stats buffer (notMoved, totalAgents)
        self.stats_buffer = self.ctx.buffer(reserve=8)  # 2 ints

        # Vertex data for square rendering
        vertices = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5]
        ], dtype='f4')
        self.square_vbo = self.ctx.buffer(vertices.tobytes())

    def upload_agents(self, agent_data: bytes) -> None:
        """Upload agent data to GPU"""
        if len(agent_data) != self.agents_buffer.size:
            raise ValueError(
                f"Agent data size mismatch: expected {self.agents_buffer.size}, "
                f"got {len(agent_data)}"
            )
        self.agents_buffer.write(agent_data)

    def upload_occupancy(self, occupancy_data: bytes) -> None:
        """Upload occupancy grid to GPU"""
        if len(occupancy_data) != self.occupancy_buffer.size:
            raise ValueError(
                f"Occupancy data size mismatch: expected {self.occupancy_buffer.size}, "
                f"got {len(occupancy_data)}"
            )
        self.occupancy_buffer.write(occupancy_data)

    def reset_stats(self) -> None:
        """Reset stats buffer"""
        stats = np.array([0, self.num_agents], dtype=np.int32)
        self.stats_buffer.write(stats.tobytes())

    def read_stats(self) -> np.ndarray:
        """Read stats from GPU"""
        return np.frombuffer(self.stats_buffer.read(), dtype=np.int32)

    def bind_for_field_compute(self) -> None:
        """Bind buffers for field generation compute shader"""
        self.field_buffer.bind_to_storage_buffer(0)

    def bind_for_agent_compute(self) -> None:
        """Bind buffers for agent movement compute shader"""
        self.agents_buffer.bind_to_storage_buffer(0)
        self.field_buffer.bind_to_storage_buffer(1)
        self.occupancy_buffer.bind_to_storage_buffer(2)
        self.stats_buffer.bind_to_storage_buffer(3)

    def create_vao(self, program: moderngl.Program) -> moderngl.VertexArray:
        """
        Create VAO for instanced rendering

        Args:
            program: Shader program to use

        Returns:
            Vertex array object
        """
        return self.ctx.vertex_array(
            program,
            [
                (self.agents_buffer, '2f 2f 1f 1f/i', 'in_position', 'in_velocity', 'in_active'),
                (self.square_vbo, '2f', 'in_vertex')
            ]
        )

    def release(self) -> None:
        """Release all buffers"""
        if self.agents_buffer:
            self.agents_buffer.release()
        if self.field_buffer:
            self.field_buffer.release()
        if self.occupancy_buffer:
            self.occupancy_buffer.release()
        if self.stats_buffer:
            self.stats_buffer.release()
        if self.square_vbo:
            self.square_vbo.release()

    def get_memory_usage_mb(self) -> float:
        """Get estimated GPU memory usage in MB"""
        total_bytes = (
            self.agents_buffer.size +
            self.field_buffer.size +
            self.occupancy_buffer.size +
            self.stats_buffer.size +
            self.square_vbo.size
        )
        return total_bytes / (1024 * 1024)
