"""
Core simulation logic - agent initialization and state management
"""

import numpy as np
from typing import Tuple


class AgentManager:
    """Manages agent data and initialization"""

    # Agent struct layout: pos(2f), velocity(2f), active(1f), padding(1f)
    AGENT_DTYPE = np.dtype([
        ('pos', np.float32, 2),
        ('velocity', np.float32, 2),
        ('active', np.float32),
        ('padding', np.float32)
    ])

    def __init__(self, num_agents: int, grid_size: int):
        """
        Initialize agent manager

        Args:
            num_agents: Number of agents to create
            grid_size: Size of the grid (grid_size x grid_size)
        """
        if num_agents < 0:
            raise ValueError(f"num_agents must be non-negative, got {num_agents}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.agents_data = np.zeros(num_agents, dtype=self.AGENT_DTYPE)

    def initialize_random_positions(self, seed: int = None) -> np.ndarray:
        """
        Initialize agents with random positions

        Args:
            seed: Random seed for reproducibility (optional)

        Returns:
            Agent data array
        """
        if seed is not None:
            np.random.seed(seed)

        # Random positions within grid
        positions = np.random.randint(
            0, self.grid_size,
            size=(self.num_agents, 2)
        ).astype(np.float32)

        self.agents_data['pos'] = positions
        self.agents_data['velocity'] = 0
        self.agents_data['active'] = 1.0

        return self.agents_data

    def initialize_grid_positions(self, spacing: int = 2) -> np.ndarray:
        """
        Initialize agents in a grid pattern (useful for testing)

        Args:
            spacing: Space between agents

        Returns:
            Agent data array
        """
        positions = []
        for y in range(0, self.grid_size, spacing):
            for x in range(0, self.grid_size, spacing):
                if len(positions) >= self.num_agents:
                    break
                positions.append([x, y])
            if len(positions) >= self.num_agents:
                break

        # Pad with zeros if we didn't fill all agents
        while len(positions) < self.num_agents:
            positions.append([0, 0])

        self.agents_data['pos'] = np.array(positions[:self.num_agents], dtype=np.float32)
        self.agents_data['velocity'] = 0
        self.agents_data['active'] = 1.0

        return self.agents_data

    def get_bytes(self) -> bytes:
        """Get agent data as bytes for GPU upload"""
        return self.agents_data.tobytes()

    def count_active(self) -> int:
        """Count active agents"""
        return int(np.sum(self.agents_data['active']))

    def get_positions(self) -> np.ndarray:
        """Get agent positions as Nx2 array"""
        return self.agents_data['pos']


class OccupancyGrid:
    """Manages occupancy grid for collision detection"""

    def __init__(self, grid_size: int):
        """
        Initialize occupancy grid

        Args:
            grid_size: Size of the grid
        """
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")

        self.grid_size = grid_size
        self.grid = np.zeros(grid_size * grid_size, dtype=np.int32)

    def mark_positions(self, positions: np.ndarray) -> None:
        """
        Mark positions as occupied

        Args:
            positions: Nx2 array of positions
        """
        self.grid.fill(0)  # Clear grid

        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                idx = y * self.grid_size + x
                self.grid[idx] = 1

    def get_bytes(self) -> bytes:
        """Get grid data as bytes for GPU upload"""
        return self.grid.tobytes()

    def is_occupied(self, x: int, y: int) -> bool:
        """Check if position is occupied"""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        idx = y * self.grid_size + x
        return self.grid[idx] != 0

    def count_occupied(self) -> int:
        """Count occupied cells"""
        return int(np.sum(self.grid != 0))


class SimulationStats:
    """Tracks simulation statistics"""

    def __init__(self):
        """Initialize stats"""
        self.frame_count = 0
        self.field_generation_count = 0
        self.total_agents_moved = 0
        self.total_agents_stuck = 0

    def update(self, agents_moved: int, agents_stuck: int) -> None:
        """Update stats for a frame"""
        self.frame_count += 1
        self.total_agents_moved += agents_moved
        self.total_agents_stuck += agents_stuck

    def field_regenerated(self) -> None:
        """Record field regeneration"""
        self.field_generation_count += 1

    def get_summary(self) -> dict:
        """Get stats summary"""
        return {
            'frames': self.frame_count,
            'field_generations': self.field_generation_count,
            'total_moved': self.total_agents_moved,
            'total_stuck': self.total_agents_stuck,
            'avg_moved_per_frame': (
                self.total_agents_moved / self.frame_count
                if self.frame_count > 0 else 0
            ),
        }

    def reset(self) -> None:
        """Reset all stats"""
        self.__init__()
