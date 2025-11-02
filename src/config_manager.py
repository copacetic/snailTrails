"""
Configuration management with validation
"""

import os
from typing import Dict, Any


class ConfigManager:
    """Manages and validates simulation configuration"""

    # Default configuration (4K widescreen)
    DEFAULTS = {
        'GRID_SIZE': 4096,
        'NUM_AGENTS': 10_000_000,
        'WINDOW_WIDTH': 3840,
        'WINDOW_HEIGHT': 2160,
        'VSYNC': True,
        'FULLSCREEN': False,
        'SHOW_FPS': True,
        'STUCK_THRESHOLD_PERCENT': 1.0,
        'FIELD_SAMPLES': 1000,
        'AGENT_SIZE': 0.6,
        'COLOR_MODE': 'velocity',
        'FIELD_WORK_GROUP_SIZE': 16,
        'AGENT_WORK_GROUP_SIZE': 256,
    }

    # Validation constraints
    CONSTRAINTS = {
        'GRID_SIZE': (16, 8192),  # Min/max grid size
        'NUM_AGENTS': (1, 100_000_000),  # Min/max agents
        'WINDOW_WIDTH': (320, 7680),  # Min/max window width
        'WINDOW_HEIGHT': (240, 4320),  # Min/max window height
        'STUCK_THRESHOLD_PERCENT': (0.0, 100.0),
        'FIELD_SAMPLES': (1, 10000),
        'AGENT_SIZE': (0.1, 2.0),
        'FIELD_WORK_GROUP_SIZE': (1, 1024),
        'AGENT_WORK_GROUP_SIZE': (1, 1024),
    }

    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize with optional config dict, using defaults for missing values"""
        self.config = self.DEFAULTS.copy()
        if config_dict:
            self.config.update(config_dict)
        self.validate()
        self._compute_derived()

    def validate(self) -> None:
        """Validate configuration values"""
        for key, (min_val, max_val) in self.CONSTRAINTS.items():
            if key in self.config:
                value = self.config[key]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{key} must be a number, got {type(value)}")
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"{key} must be between {min_val} and {max_val}, got {value}"
                    )

        # Grid size must be power of 2 or reasonable for work groups
        grid_size = self.config['GRID_SIZE']
        work_group_size = self.config['FIELD_WORK_GROUP_SIZE']
        if grid_size % work_group_size != 0:
            raise ValueError(
                f"GRID_SIZE ({grid_size}) should be divisible by "
                f"FIELD_WORK_GROUP_SIZE ({work_group_size})"
            )

        # Validate color mode
        valid_modes = ['velocity', 'speed', 'random']
        if self.config['COLOR_MODE'] not in valid_modes:
            raise ValueError(
                f"COLOR_MODE must be one of {valid_modes}, "
                f"got {self.config['COLOR_MODE']}"
            )

    def _compute_derived(self) -> None:
        """Compute derived configuration values"""
        self.config['STUCK_THRESHOLD'] = int(
            self.config['NUM_AGENTS'] * (self.config['STUCK_THRESHOLD_PERCENT'] / 100.0)
        )

        # Memory estimates (MB)
        self.config['AGENT_MEMORY_MB'] = (
            self.config['NUM_AGENTS'] * 24
        ) / (1024 * 1024)
        self.config['FIELD_MEMORY_MB'] = (
            self.config['GRID_SIZE'] ** 2 * 8
        ) / (1024 * 1024)
        self.config['GRID_MEMORY_MB'] = (
            self.config['GRID_SIZE'] ** 2 * 4
        ) / (1024 * 1024)
        self.config['TOTAL_MEMORY_MB'] = (
            self.config['AGENT_MEMORY_MB']
            + self.config['FIELD_MEMORY_MB']
            + self.config['GRID_MEMORY_MB']
        )

    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def __getitem__(self, key: str):
        """Allow dict-style access"""
        return self.config[key]

    def print_summary(self) -> None:
        """Print configuration summary"""
        print("=" * 70)
        print("Configuration:")
        print(f"  Grid: {self['GRID_SIZE']}x{self['GRID_SIZE']} "
              f"({self['GRID_SIZE']**2:,} cells)")
        print(f"  Agents: {self['NUM_AGENTS']:,}")
        print(f"  Est. VRAM: ~{self['TOTAL_MEMORY_MB']:.1f} MB")
        print(f"  Window: {self['WINDOW_WIDTH']}x{self['WINDOW_HEIGHT']}")
        print(f"  Field samples: {self['FIELD_SAMPLES']}")
        print("=" * 70)


def load_config_from_file(filepath: str = 'config.py') -> ConfigManager:
    """Load configuration from Python file"""
    config_dict = {}

    if os.path.exists(filepath):
        # Execute config file and extract uppercase variables
        with open(filepath, 'r') as f:
            exec_globals = {}
            exec(f.read(), exec_globals)
            config_dict = {
                k: v for k, v in exec_globals.items()
                if k.isupper() and not k.startswith('_')
            }

    return ConfigManager(config_dict)
