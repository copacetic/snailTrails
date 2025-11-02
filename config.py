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
#   2048  - Large (recommended for 1080p)
#   4096  - Extreme (perfect for 4K displays!)
GRID_SIZE = 4096

# Number of agents
# RTX 4090 recommendations:
#   100,000    - Warm-up
#   1,000,000  - Good starting point
#   5,000,000  - Balanced
#   10,000,000 - Standard (10M agents)
#   20,000,000 - High (20M agents)
#   50,000,000 - Extreme (50M agents - ~12GB VRAM)
#   100,000,000 - INSANE MODE! (100M agents - ~18GB VRAM) 🔥
NUM_AGENTS = 50_000_000  # EXTREME MODE ENABLED!

# ===========================================
# DISPLAY SETTINGS
# ===========================================

# Window resolution (4K widescreen default)
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160

# Vsync (limits FPS to monitor refresh rate)
VSYNC = True

# Fullscreen mode (recommended for 4K displays)
FULLSCREEN = False

# Show FPS counter in title
SHOW_FPS = True

# Performance monitoring
SHOW_DETAILED_STATS = True  # Show frame time, GPU usage stats
BENCHMARK_MODE = False       # Run for 300 frames then show average FPS
TARGET_FPS = 60             # Target framerate

# ===========================================
# SIMULATION BEHAVIOR
# ===========================================

# Regenerate vector field when % of agents stuck
# Lower = more frequent field changes
# Higher = agents follow field longer
STUCK_THRESHOLD_PERCENT = 1.0  # Regenerate when <1% moving

# Vector field complexity
# Higher = more detailed patterns, slower generation
# 4K displays benefit from higher sample counts
FIELD_SAMPLES = 2000  # EXTREME: 2000 samples for ultra-smooth fields

# ===========================================
# ADVANCED SETTINGS
# ===========================================

# Compute shader work group sizes
# Optimized for RTX 4090!
FIELD_WORK_GROUP_SIZE = 16   # 16x16 for field generation
AGENT_WORK_GROUP_SIZE = 512  # 512 threads for agent updates (2x default!)

# Advanced rendering options
ENABLE_MOTION_BLUR = False   # Smooth trails (experimental)
ENABLE_GLOW_EFFECT = False   # Agents glow based on speed
PARTICLE_DENSITY = 1.0       # Density multiplier (0.5 = sparse, 2.0 = dense)

# Agent render size multiplier
# 1.0 = agents fill grid cells
# 0.8 = agents slightly smaller
# 0.6 = good for 4K displays
# 0.5 = tiny agents (more detail)
# 0.3 = microscopic (extreme detail) 🔬
AGENT_SIZE = 0.4  # Extreme detail mode

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
# AGENT_SIZE = 0.8
# FIELD_SAMPLES = 200
# AGENT_WORK_GROUP_SIZE = 256

# # PRESET: 1080p Balanced
# GRID_SIZE = 1024
# NUM_AGENTS = 1_000_000
# WINDOW_WIDTH = 1920
# WINDOW_HEIGHT = 1080
# AGENT_SIZE = 0.8
# FIELD_SAMPLES = 500
# AGENT_WORK_GROUP_SIZE = 256

# # PRESET: 1080p High Performance
# GRID_SIZE = 2048
# NUM_AGENTS = 10_000_000
# WINDOW_WIDTH = 1920
# WINDOW_HEIGHT = 1080
# AGENT_SIZE = 0.7
# FIELD_SAMPLES = 500
# AGENT_WORK_GROUP_SIZE = 512

# # PRESET: 4K Balanced
# GRID_SIZE = 4096
# NUM_AGENTS = 10_000_000
# WINDOW_WIDTH = 3840
# WINDOW_HEIGHT = 2160
# AGENT_SIZE = 0.6
# FIELD_SAMPLES = 1000
# AGENT_WORK_GROUP_SIZE = 256

# # PRESET: 4K High Performance
# GRID_SIZE = 4096
# NUM_AGENTS = 20_000_000
# WINDOW_WIDTH = 3840
# WINDOW_HEIGHT = 2160
# AGENT_SIZE = 0.5
# FIELD_SAMPLES = 1500
# AGENT_WORK_GROUP_SIZE = 512

# # PRESET: 4K EXTREME (50M agents) 🔥
# GRID_SIZE = 4096
# NUM_AGENTS = 50_000_000
# WINDOW_WIDTH = 3840
# WINDOW_HEIGHT = 2160
# AGENT_SIZE = 0.4
# FIELD_SAMPLES = 2000
# AGENT_WORK_GROUP_SIZE = 512
# SHOW_DETAILED_STATS = True

# # PRESET: INSANE MODE (100M agents) 💀
# GRID_SIZE = 4096
# NUM_AGENTS = 100_000_000
# WINDOW_WIDTH = 3840
# WINDOW_HEIGHT = 2160
# AGENT_SIZE = 0.3
# FIELD_SAMPLES = 2000
# AGENT_WORK_GROUP_SIZE = 1024
# SHOW_DETAILED_STATS = True
# BENCHMARK_MODE = True

# # PRESET: ABSOLUTE MAXIMUM (push to the limit!)
# GRID_SIZE = 8192
# NUM_AGENTS = 100_000_000
# WINDOW_WIDTH = 3840
# WINDOW_HEIGHT = 2160
# AGENT_SIZE = 0.2
# FIELD_SAMPLES = 3000
# AGENT_WORK_GROUP_SIZE = 1024
# SHOW_DETAILED_STATS = True

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
