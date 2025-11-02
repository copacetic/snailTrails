"""
Configuration file for GPU-accelerated Snail Trails
Edit these values to scale the simulation
"""

# ===========================================
# SIMULATION SCALE
# ===========================================

# Grid size (NxN cells)
# RTX 4090 recommendations:
#   512   - Small (fast testing)
#   1024  - Medium (good balance)
#   2048  - Large (recommended)
#   4096  - Extreme (max detail)
GRID_SIZE = 2048

# Number of agents
# RTX 4090 recommendations:
#   100,000    - Warm-up
#   1,000,000  - Good starting point
#   5,000,000  - Balanced
#   10,000,000 - RECOMMENDED (10M agents!)
#   20,000,000 - Ultra scale
#   50,000,000 - Extreme (uses ~12GB VRAM)
NUM_AGENTS = 10_000_000

# ===========================================
# DISPLAY SETTINGS
# ===========================================

# Window resolution
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

# Vsync (limits FPS to monitor refresh rate)
VSYNC = True

# Show FPS counter in title
SHOW_FPS = True

# ===========================================
# SIMULATION BEHAVIOR
# ===========================================

# Regenerate vector field when % of agents stuck
# Lower = more frequent field changes
# Higher = agents follow field longer
STUCK_THRESHOLD_PERCENT = 1.0  # Regenerate when <1% moving

# Vector field complexity
# Higher = more detailed patterns, slower generation
FIELD_SAMPLES = 500  # Per-cell samples (500 is good balance)

# ===========================================
# ADVANCED SETTINGS
# ===========================================

# Compute shader work group sizes
# Only change if you know what you're doing!
FIELD_WORK_GROUP_SIZE = 16   # 16x16 for field generation
AGENT_WORK_GROUP_SIZE = 256  # 256 threads for agent updates

# Agent render size multiplier
# 1.0 = agents fill grid cells
# 0.8 = agents slightly smaller (default)
# 0.5 = tiny agents
AGENT_SIZE = 0.8

# Color mode
# 'velocity' - Rainbow based on direction (default)
# 'speed' - Grayscale based on speed
# 'random' - Random per agent
COLOR_MODE = 'velocity'

# ===========================================
# PRESETS (uncomment to use)
# ===========================================

# # PRESET: Quick Test (fast, for debugging)
# GRID_SIZE = 512
# NUM_AGENTS = 100_000
# WINDOW_WIDTH = 1280
# WINDOW_HEIGHT = 720

# # PRESET: Balanced (good for most GPUs)
# GRID_SIZE = 1024
# NUM_AGENTS = 1_000_000
# WINDOW_WIDTH = 1920
# WINDOW_HEIGHT = 1080

# # PRESET: RTX 4090 Full Power (RECOMMENDED)
# GRID_SIZE = 2048
# NUM_AGENTS = 10_000_000
# WINDOW_WIDTH = 1920
# WINDOW_HEIGHT = 1080

# # PRESET: Extreme Scale (RTX 4090 stress test)
# GRID_SIZE = 4096
# NUM_AGENTS = 50_000_000
# WINDOW_WIDTH = 3840
# WINDOW_HEIGHT = 2160

# ===========================================
# CALCULATED VALUES (don't edit)
# ===========================================

# Agent update threshold for field regeneration
STUCK_THRESHOLD = int(NUM_AGENTS * (STUCK_THRESHOLD_PERCENT / 100.0))

# Memory estimates (MB)
AGENT_MEMORY_MB = (NUM_AGENTS * 24) / (1024 * 1024)  # 24 bytes per agent
FIELD_MEMORY_MB = (GRID_SIZE * GRID_SIZE * 8) / (1024 * 1024)  # 8 bytes per cell
GRID_MEMORY_MB = (GRID_SIZE * GRID_SIZE * 4) / (1024 * 1024)  # 4 bytes per cell
TOTAL_MEMORY_MB = AGENT_MEMORY_MB + FIELD_MEMORY_MB + GRID_MEMORY_MB

# Print configuration on import
if __name__ != '__main__':
    print("=" * 60)
    print(f"Configuration Loaded:")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE} ({GRID_SIZE*GRID_SIZE:,} cells)")
    print(f"  Agents: {NUM_AGENTS:,}")
    print(f"  Est. VRAM: ~{TOTAL_MEMORY_MB:.1f} MB")
    print(f"  Window: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    print("=" * 60)
