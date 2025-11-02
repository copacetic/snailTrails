"""
Additional smoke tests for real-world scenarios
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_manager import ConfigManager, load_config_from_file
from src.simulation import AgentManager, OccupancyGrid, SimulationStats
from src.shaders import ShaderManager


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    def test_full_cpu_pipeline(self):
        """Test complete CPU-side pipeline"""
        # Setup
        config = ConfigManager({
            'GRID_SIZE': 128,
            'NUM_AGENTS': 1000,
            'STUCK_THRESHOLD_PERCENT': 5.0
        })

        # Initialize agents
        agent_mgr = AgentManager(
            num_agents=config['NUM_AGENTS'],
            grid_size=config['GRID_SIZE']
        )
        agent_mgr.initialize_random_positions(seed=123)

        # Create occupancy grid
        occ_grid = OccupancyGrid(config['GRID_SIZE'])
        occ_grid.mark_positions(agent_mgr.get_positions())

        # Verify data
        assert agent_mgr.count_active() == 1000
        assert occ_grid.count_occupied() <= 1000  # May have collisions

        # Get data for GPU upload
        agent_bytes = agent_mgr.get_bytes()
        occ_bytes = occ_grid.get_bytes()

        assert len(agent_bytes) == 1000 * 24  # 24 bytes per agent
        assert len(occ_bytes) == 128 * 128 * 4  # 4 bytes per cell

    def test_large_scale_initialization(self):
        """Test initialization with large agent count"""
        # 1 million agents
        agent_mgr = AgentManager(num_agents=1_000_000, grid_size=2048)
        agent_mgr.initialize_random_positions(seed=42)

        assert agent_mgr.count_active() == 1_000_000

        # Verify positions are within bounds
        positions = agent_mgr.get_positions()
        assert np.all(positions >= 0)
        assert np.all(positions < 2048)

    def test_stats_tracking_workflow(self):
        """Test statistics tracking through multiple frames"""
        stats = SimulationStats()

        # Simulate 10 frames
        for frame in range(10):
            moved = 900 - (frame * 10)  # Decreasing movement
            stuck = 100 + (frame * 10)  # Increasing stuck
            stats.update(agents_moved=moved, agents_stuck=stuck)

        # Check accumulated stats
        summary = stats.get_summary()
        assert summary['frames'] == 10
        assert summary['total_moved'] == 8550  # Sum of arithmetic sequence
        assert summary['total_stuck'] == 1450

    def test_config_memory_estimates_scale(self):
        """Test memory estimates at different scales"""
        # Small config
        small = ConfigManager({'GRID_SIZE': 512, 'NUM_AGENTS': 10_000})
        assert small['TOTAL_MEMORY_MB'] < 10

        # Medium config
        medium = ConfigManager({'GRID_SIZE': 1024, 'NUM_AGENTS': 1_000_000})
        assert 10 < medium['TOTAL_MEMORY_MB'] < 100

        # Large config
        large = ConfigManager({'GRID_SIZE': 2048, 'NUM_AGENTS': 10_000_000})
        assert 100 < large['TOTAL_MEMORY_MB'] < 1000

    def test_occupancy_collision_handling(self):
        """Test occupancy grid with many collisions"""
        grid = OccupancyGrid(grid_size=10)

        # Place many agents at same locations
        positions = np.array([[5, 5]] * 100, dtype=np.float32)
        grid.mark_positions(positions)

        # Should only count once
        assert grid.count_occupied() == 1
        assert grid.is_occupied(5, 5)

    def test_agent_position_boundaries(self):
        """Test agents at grid boundaries"""
        agent_mgr = AgentManager(num_agents=4, grid_size=100)

        # Place agents at corners
        agent_mgr.agents_data['pos'] = np.array([
            [0, 0],      # Bottom-left
            [99, 0],     # Bottom-right
            [0, 99],     # Top-left
            [99, 99]     # Top-right
        ], dtype=np.float32)
        agent_mgr.agents_data['active'] = 1.0

        # All should be valid
        assert agent_mgr.count_active() == 4

        positions = agent_mgr.get_positions()
        assert np.all(positions >= 0)
        assert np.all(positions < 100)

    def test_shader_file_completeness(self):
        """Test all shader files have required content"""
        shader_dir = os.path.join(os.path.dirname(__file__), '..', 'shaders')

        # Required shader files and their key content
        required_shaders = {
            'field_compute.glsl': [
                '#version 430',
                'layout(local_size_x = 16, local_size_y = 16)',
                'uniform int gridSize',
                'uniform float time',
                'uniform int samples'
            ],
            'agent_compute.glsl': [
                '#version 430',
                'layout(local_size_x = 512)',  # EXTREME mode work group size
                'uniform int gridSize',
                'uniform int numAgents',
                'atomicCompSwap'
            ],
            'vertex.glsl': [
                '#version 430',
                'uniform mat4 projection',
                'in vec2 in_position',
                'in vec2 in_velocity'
            ],
            'fragment.glsl': [
                '#version 430',
                'out vec4 fragColor'
            ]
        }

        for shader_file, required_content in required_shaders.items():
            filepath = os.path.join(shader_dir, shader_file)
            assert os.path.exists(filepath), f"Missing {shader_file}"

            with open(filepath, 'r') as f:
                content = f.read()

            for required in required_content:
                assert required in content, \
                    f"{shader_file} missing required content: {required}"

    def test_data_type_consistency(self):
        """Test that data types are consistent across pipeline"""
        agent_mgr = AgentManager(num_agents=100, grid_size=512)
        agent_mgr.initialize_random_positions()

        # Check dtype of positions
        positions = agent_mgr.get_positions()
        assert positions.dtype == np.float32

        # Check dtype of agent data
        assert agent_mgr.agents_data['pos'].dtype == np.float32
        assert agent_mgr.agents_data['velocity'].dtype == np.float32
        assert agent_mgr.agents_data['active'].dtype == np.float32

        # Check occupancy grid dtype
        occ_grid = OccupancyGrid(512)
        assert occ_grid.grid.dtype == np.int32

    def test_configuration_from_file(self):
        """Test loading configuration from actual config.py file"""
        # Check if config.py exists
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')

        if os.path.exists(config_path):
            config = load_config_from_file(config_path)

            # Should have loaded values
            assert config['GRID_SIZE'] > 0
            assert config['NUM_AGENTS'] > 0

            # Should have computed derived values
            assert 'STUCK_THRESHOLD' in config.config
            assert 'TOTAL_MEMORY_MB' in config.config

    def test_agent_data_serialization(self):
        """Test agent data can be serialized and matches expected format"""
        agent_mgr = AgentManager(num_agents=10, grid_size=64)
        agent_mgr.initialize_random_positions(seed=99)

        # Get bytes
        data_bytes = agent_mgr.get_bytes()

        # Should be correct size
        assert len(data_bytes) == 10 * 24  # 10 agents * 24 bytes

        # Should be able to reconstruct
        reconstructed = np.frombuffer(data_bytes, dtype=agent_mgr.AGENT_DTYPE)
        assert len(reconstructed) == 10

        # Data should match
        np.testing.assert_array_equal(
            reconstructed['pos'],
            agent_mgr.agents_data['pos']
        )

    def test_work_group_calculations(self):
        """Test work group size calculations are correct"""
        test_cases = [
            (512, 16, 32),    # 512/16 = 32 groups
            (1024, 16, 64),   # 1024/16 = 64 groups
            (2048, 16, 128),  # 2048/16 = 128 groups
        ]

        for grid_size, work_group_size, expected_groups in test_cases:
            config = ConfigManager({
                'GRID_SIZE': grid_size,
                'FIELD_WORK_GROUP_SIZE': work_group_size
            })

            groups = (config['GRID_SIZE'] + work_group_size - 1) // work_group_size
            assert groups == expected_groups

    def test_stats_edge_cases(self):
        """Test statistics with edge cases"""
        stats = SimulationStats()

        # All agents stuck
        stats.update(agents_moved=0, agents_stuck=10000)
        assert stats.total_agents_stuck == 10000

        # All agents moving
        stats.update(agents_moved=10000, agents_stuck=0)
        assert stats.total_agents_moved == 10000

        summary = stats.get_summary()
        assert summary['frames'] == 2
        assert summary['avg_moved_per_frame'] == 5000

    def test_modular_version_imports(self):
        """Test that modular version can be imported"""
        # Try to import the modular version
        import sys
        import os

        modular_path = os.path.join(os.path.dirname(__file__), '..', 'snail_trails_modular.py')
        assert os.path.exists(modular_path), "snail_trails_modular.py not found"

        # Check it has the main class
        with open(modular_path, 'r') as f:
            content = f.read()
            assert 'class SnailTrailsGPU' in content
            assert 'from src.config_manager import' in content
            assert 'from src.simulation import' in content
            assert 'from src.gpu_buffers import' in content
            assert 'from src.shaders import' in content


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_positions_dont_crash(self):
        """Test that invalid positions are handled gracefully"""
        grid = OccupancyGrid(grid_size=100)

        # Include clearly invalid positions
        positions = np.array([
            [50, 50],      # Valid
            [-100, -100],  # Invalid
            [1000, 1000],  # Invalid
            [50, 200],     # Partially invalid
        ], dtype=np.float32)

        # Should not crash
        grid.mark_positions(positions)

        # Only valid position should be marked
        assert grid.is_occupied(50, 50)
        assert grid.count_occupied() == 1

    def test_empty_agent_operations(self):
        """Test operations with zero agents"""
        agent_mgr = AgentManager(num_agents=0, grid_size=512)
        agent_mgr.initialize_random_positions()

        assert agent_mgr.count_active() == 0
        assert len(agent_mgr.get_positions()) == 0
        assert len(agent_mgr.get_bytes()) == 0

    def test_minimum_config_values(self):
        """Test configuration with minimum valid values"""
        config = ConfigManager({
            'GRID_SIZE': 16,
            'NUM_AGENTS': 1,
            'WINDOW_WIDTH': 320,
            'WINDOW_HEIGHT': 240
        })

        # Should work with minimal values
        agent_mgr = AgentManager(
            num_agents=config['NUM_AGENTS'],
            grid_size=config['GRID_SIZE']
        )
        agent_mgr.initialize_random_positions()

        assert agent_mgr.count_active() == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
