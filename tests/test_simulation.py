"""
Tests for simulation logic
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation import AgentManager, OccupancyGrid, SimulationStats


class TestAgentManager:
    """Test agent management"""

    def test_initialization(self):
        """Test agent manager initialization"""
        manager = AgentManager(num_agents=100, grid_size=512)
        assert manager.num_agents == 100
        assert manager.grid_size == 512
        assert len(manager.agents_data) == 100

    def test_invalid_initialization(self):
        """Test invalid initialization parameters"""
        # Negative agents
        with pytest.raises(ValueError, match="must be non-negative"):
            AgentManager(num_agents=-1, grid_size=512)

        # Zero grid size
        with pytest.raises(ValueError, match="must be positive"):
            AgentManager(num_agents=100, grid_size=0)

    def test_random_positions(self):
        """Test random position initialization"""
        manager = AgentManager(num_agents=100, grid_size=512)
        data = manager.initialize_random_positions(seed=42)

        # All agents should be active
        assert np.all(data['active'] == 1.0)

        # Positions should be within bounds
        assert np.all(data['pos'] >= 0)
        assert np.all(data['pos'] < 512)

        # Velocities should be zero initially
        assert np.all(data['velocity'] == 0)

    def test_random_positions_reproducible(self):
        """Test that random positions are reproducible with seed"""
        manager1 = AgentManager(num_agents=100, grid_size=512)
        data1 = manager1.initialize_random_positions(seed=42)

        manager2 = AgentManager(num_agents=100, grid_size=512)
        data2 = manager2.initialize_random_positions(seed=42)

        # Should be identical
        np.testing.assert_array_equal(data1['pos'], data2['pos'])

    def test_grid_positions(self):
        """Test grid pattern initialization"""
        manager = AgentManager(num_agents=100, grid_size=512)
        data = manager.initialize_grid_positions(spacing=10)

        # All agents should be active
        assert np.all(data['active'] == 1.0)

        # Positions should be on grid
        positions = data['pos']
        unique_positions = np.unique(positions, axis=0)

        # Should have multiple unique positions
        assert len(unique_positions) > 1

    def test_grid_positions_overflow(self):
        """Test grid initialization with more agents than grid spaces"""
        # Request more agents than can fit
        manager = AgentManager(num_agents=1000, grid_size=10)
        data = manager.initialize_grid_positions(spacing=5)

        # Should still create 1000 agents (some at same position)
        assert len(data) == 1000
        assert manager.count_active() == 1000

    def test_get_bytes(self):
        """Test getting agent data as bytes"""
        manager = AgentManager(num_agents=10, grid_size=512)
        manager.initialize_random_positions()

        byte_data = manager.get_bytes()

        # Should be correct size: 10 agents * 24 bytes per agent
        assert len(byte_data) == 10 * 24

    def test_count_active(self):
        """Test counting active agents"""
        manager = AgentManager(num_agents=100, grid_size=512)
        manager.initialize_random_positions()

        # Initially all active
        assert manager.count_active() == 100

        # Deactivate some
        manager.agents_data['active'][:50] = 0.0
        assert manager.count_active() == 50

    def test_get_positions(self):
        """Test getting positions array"""
        manager = AgentManager(num_agents=100, grid_size=512)
        manager.initialize_random_positions(seed=42)

        positions = manager.get_positions()

        # Should be Nx2 array
        assert positions.shape == (100, 2)

        # Should match agent data
        np.testing.assert_array_equal(positions, manager.agents_data['pos'])

    def test_zero_agents(self):
        """Test edge case with zero agents"""
        manager = AgentManager(num_agents=0, grid_size=512)
        data = manager.initialize_random_positions()

        assert len(data) == 0
        assert manager.count_active() == 0


class TestOccupancyGrid:
    """Test occupancy grid"""

    def test_initialization(self):
        """Test occupancy grid initialization"""
        grid = OccupancyGrid(grid_size=512)
        assert grid.grid_size == 512
        assert len(grid.grid) == 512 * 512
        assert np.all(grid.grid == 0)

    def test_invalid_initialization(self):
        """Test invalid grid size"""
        with pytest.raises(ValueError, match="must be positive"):
            OccupancyGrid(grid_size=0)

        with pytest.raises(ValueError, match="must be positive"):
            OccupancyGrid(grid_size=-10)

    def test_mark_positions(self):
        """Test marking positions as occupied"""
        grid = OccupancyGrid(grid_size=100)

        positions = np.array([
            [10, 20],
            [30, 40],
            [50, 60]
        ], dtype=np.float32)

        grid.mark_positions(positions)

        # Marked positions should be occupied
        assert grid.is_occupied(10, 20)
        assert grid.is_occupied(30, 40)
        assert grid.is_occupied(50, 60)

        # Other positions should be free
        assert not grid.is_occupied(0, 0)
        assert not grid.is_occupied(99, 99)

    def test_mark_positions_out_of_bounds(self):
        """Test marking positions outside grid bounds"""
        grid = OccupancyGrid(grid_size=100)

        # Include out-of-bounds positions
        positions = np.array([
            [10, 20],
            [200, 200],  # Out of bounds
            [-5, -5]     # Out of bounds
        ], dtype=np.float32)

        # Should not crash
        grid.mark_positions(positions)

        # Valid position should be marked
        assert grid.is_occupied(10, 20)

        # Should only have 1 occupied cell
        assert grid.count_occupied() == 1

    def test_is_occupied_bounds(self):
        """Test is_occupied with out-of-bounds coordinates"""
        grid = OccupancyGrid(grid_size=100)

        # Out of bounds should return False
        assert not grid.is_occupied(-1, 0)
        assert not grid.is_occupied(0, -1)
        assert not grid.is_occupied(100, 0)
        assert not grid.is_occupied(0, 100)

    def test_count_occupied(self):
        """Test counting occupied cells"""
        grid = OccupancyGrid(grid_size=100)

        assert grid.count_occupied() == 0

        positions = np.array([[i, i] for i in range(50)], dtype=np.float32)
        grid.mark_positions(positions)

        assert grid.count_occupied() == 50

    def test_clear_and_remark(self):
        """Test that marking positions clears previous marks"""
        grid = OccupancyGrid(grid_size=100)

        # Mark first set
        positions1 = np.array([[10, 10]], dtype=np.float32)
        grid.mark_positions(positions1)
        assert grid.is_occupied(10, 10)

        # Mark second set (should clear first)
        positions2 = np.array([[20, 20]], dtype=np.float32)
        grid.mark_positions(positions2)

        assert not grid.is_occupied(10, 10)
        assert grid.is_occupied(20, 20)
        assert grid.count_occupied() == 1

    def test_get_bytes(self):
        """Test getting grid data as bytes"""
        grid = OccupancyGrid(grid_size=100)
        byte_data = grid.get_bytes()

        # Should be correct size: 100*100 * 4 bytes per int
        assert len(byte_data) == 100 * 100 * 4

    def test_duplicate_positions(self):
        """Test marking same position multiple times"""
        grid = OccupancyGrid(grid_size=100)

        # Multiple agents at same position
        positions = np.array([
            [10, 10],
            [10, 10],
            [10, 10]
        ], dtype=np.float32)

        grid.mark_positions(positions)

        # Should only count once
        assert grid.count_occupied() == 1


class TestSimulationStats:
    """Test simulation statistics"""

    def test_initialization(self):
        """Test stats initialization"""
        stats = SimulationStats()
        assert stats.frame_count == 0
        assert stats.field_generation_count == 0
        assert stats.total_agents_moved == 0
        assert stats.total_agents_stuck == 0

    def test_update(self):
        """Test updating stats"""
        stats = SimulationStats()

        stats.update(agents_moved=100, agents_stuck=50)

        assert stats.frame_count == 1
        assert stats.total_agents_moved == 100
        assert stats.total_agents_stuck == 50

        stats.update(agents_moved=80, agents_stuck=70)

        assert stats.frame_count == 2
        assert stats.total_agents_moved == 180
        assert stats.total_agents_stuck == 120

    def test_field_regeneration(self):
        """Test field regeneration tracking"""
        stats = SimulationStats()

        assert stats.field_generation_count == 0

        stats.field_regenerated()
        assert stats.field_generation_count == 1

        stats.field_regenerated()
        stats.field_regenerated()
        assert stats.field_generation_count == 3

    def test_get_summary(self):
        """Test getting stats summary"""
        stats = SimulationStats()

        stats.update(100, 50)
        stats.update(80, 70)
        stats.field_regenerated()

        summary = stats.get_summary()

        assert summary['frames'] == 2
        assert summary['field_generations'] == 1
        assert summary['total_moved'] == 180
        assert summary['total_stuck'] == 120
        assert summary['avg_moved_per_frame'] == 90.0

    def test_reset(self):
        """Test resetting stats"""
        stats = SimulationStats()

        stats.update(100, 50)
        stats.field_regenerated()

        stats.reset()

        assert stats.frame_count == 0
        assert stats.field_generation_count == 0
        assert stats.total_agents_moved == 0
        assert stats.total_agents_stuck == 0

    def test_avg_with_zero_frames(self):
        """Test average calculation with zero frames"""
        stats = SimulationStats()

        summary = stats.get_summary()
        assert summary['avg_moved_per_frame'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
