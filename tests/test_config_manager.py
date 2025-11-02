"""
Tests for configuration management
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_manager import ConfigManager


class TestConfigManager:
    """Test configuration management"""

    def test_default_config(self):
        """Test default configuration loads correctly"""
        config = ConfigManager()
        assert config['GRID_SIZE'] == 2048
        assert config['NUM_AGENTS'] == 10_000_000
        assert config['WINDOW_WIDTH'] == 1920
        assert config['WINDOW_HEIGHT'] == 1080

    def test_custom_config(self):
        """Test custom configuration"""
        custom = {'GRID_SIZE': 512, 'NUM_AGENTS': 1000}
        config = ConfigManager(custom)
        assert config['GRID_SIZE'] == 512
        assert config['NUM_AGENTS'] == 1000
        # Defaults still apply
        assert config['WINDOW_WIDTH'] == 1920

    def test_validation_grid_size(self):
        """Test grid size validation"""
        # Too small
        with pytest.raises(ValueError, match="GRID_SIZE must be between"):
            ConfigManager({'GRID_SIZE': 8})

        # Too large
        with pytest.raises(ValueError, match="GRID_SIZE must be between"):
            ConfigManager({'GRID_SIZE': 10000})

    def test_validation_num_agents(self):
        """Test agent count validation"""
        # Negative
        with pytest.raises(ValueError, match="NUM_AGENTS must be between"):
            ConfigManager({'NUM_AGENTS': -1})

        # Too many
        with pytest.raises(ValueError, match="NUM_AGENTS must be between"):
            ConfigManager({'NUM_AGENTS': 200_000_000})

    def test_validation_work_group_alignment(self):
        """Test work group size alignment"""
        # Grid not divisible by work group size
        with pytest.raises(ValueError, match="should be divisible"):
            ConfigManager({'GRID_SIZE': 1000, 'FIELD_WORK_GROUP_SIZE': 16})

        # Valid alignment
        config = ConfigManager({'GRID_SIZE': 1024, 'FIELD_WORK_GROUP_SIZE': 16})
        assert config['GRID_SIZE'] == 1024

    def test_validation_color_mode(self):
        """Test color mode validation"""
        # Invalid mode
        with pytest.raises(ValueError, match="COLOR_MODE must be one of"):
            ConfigManager({'COLOR_MODE': 'invalid'})

        # Valid modes
        for mode in ['velocity', 'speed', 'random']:
            config = ConfigManager({'COLOR_MODE': mode})
            assert config['COLOR_MODE'] == mode

    def test_derived_values(self):
        """Test derived configuration values are computed"""
        config = ConfigManager({'NUM_AGENTS': 1000, 'STUCK_THRESHOLD_PERCENT': 10.0})

        # Stuck threshold should be computed
        assert config['STUCK_THRESHOLD'] == 100  # 10% of 1000

        # Memory estimates should exist
        assert 'AGENT_MEMORY_MB' in config.config
        assert 'FIELD_MEMORY_MB' in config.config
        assert 'TOTAL_MEMORY_MB' in config.config

    def test_memory_estimates(self):
        """Test memory estimation calculations"""
        config = ConfigManager({'NUM_AGENTS': 1000, 'GRID_SIZE': 512})

        # Agents: 1000 * 24 bytes = 24000 bytes = ~0.023 MB
        assert config['AGENT_MEMORY_MB'] < 0.1

        # Field: 512*512 * 8 bytes = 2MB
        assert 1.9 < config['FIELD_MEMORY_MB'] < 2.1

        # Grid: 512*512 * 4 bytes = 1MB
        assert 0.9 < config['GRID_MEMORY_MB'] < 1.1

    def test_get_method(self):
        """Test get method with defaults"""
        config = ConfigManager()

        # Existing key
        assert config.get('GRID_SIZE') == 2048

        # Non-existing key with default
        assert config.get('NONEXISTENT', 42) == 42

        # Non-existing key without default
        assert config.get('NONEXISTENT') is None

    def test_dict_access(self):
        """Test dictionary-style access"""
        config = ConfigManager({'GRID_SIZE': 1024})

        # Should work like a dict
        assert config['GRID_SIZE'] == 1024

        # Should raise KeyError for missing keys
        with pytest.raises(KeyError):
            _ = config['NONEXISTENT']

    def test_bounds_validation(self):
        """Test edge cases for bounds"""
        # Minimum valid values
        config = ConfigManager({
            'GRID_SIZE': 16,
            'NUM_AGENTS': 1,
            'WINDOW_WIDTH': 320,
            'WINDOW_HEIGHT': 240,
            'STUCK_THRESHOLD_PERCENT': 0.0,
            'FIELD_SAMPLES': 1,
            'AGENT_SIZE': 0.1
        })
        assert config['GRID_SIZE'] == 16

        # Maximum valid values
        config = ConfigManager({
            'GRID_SIZE': 8192,
            'NUM_AGENTS': 100_000_000,
            'STUCK_THRESHOLD_PERCENT': 100.0,
            'FIELD_SAMPLES': 10000,
            'AGENT_SIZE': 2.0
        })
        assert config['GRID_SIZE'] == 8192

    def test_type_validation(self):
        """Test that non-numeric values are rejected"""
        with pytest.raises(ValueError, match="must be a number"):
            ConfigManager({'GRID_SIZE': "not a number"})

        with pytest.raises(ValueError, match="must be a number"):
            ConfigManager({'NUM_AGENTS': None})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
