# 🖥️ 4K Widescreen Setup Guide

## Your System is Now Optimized for 4K!

The default configuration has been set for **3840x2160 (4K UHD)** displays.

---

## Current 4K Settings ✅

```python
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160
GRID_SIZE = 4096        # 16.7M cells for crisp detail
NUM_AGENTS = 10_000_000  # 10 million agents
AGENT_SIZE = 0.6        # Smaller for better 4K detail
FIELD_SAMPLES = 1000    # More samples for smoother patterns
```

### Memory Usage
- **Estimated VRAM:** ~420 MB
- **Leaves plenty of room** on your 24GB RTX 4090
- You can scale up to 50M agents if you want!

---

## Quick Start

### 1. Run with Default 4K Settings
```bash
python snail_trails_modular.py
```

This will run at **3840x2160** with **10 million agents** on a **4096x4096 grid**.

### 2. Enable Fullscreen (Optional)
Edit `config.py`:
```python
FULLSCREEN = True
```

---

## Performance Expectations on RTX 4090

| Configuration | Agents | Grid | Expected FPS | VRAM |
|---------------|--------|------|--------------|------|
| **Current (4K)** | 10M | 4096² | 60+ | ~420 MB |
| 4K Ultra | 20M | 4096² | 45+ | ~840 MB |
| 4K Extreme | 50M | 4096² | 20-30 | ~1.9 GB |

---

## Visual Quality Improvements for 4K

### Smaller Agents = More Detail
```python
AGENT_SIZE = 0.6  # Default
AGENT_SIZE = 0.5  # Even more detail
AGENT_SIZE = 0.4  # Maximum detail (tiny agents)
```

### More Field Samples = Smoother Patterns
```python
FIELD_SAMPLES = 1000  # Default (good balance)
FIELD_SAMPLES = 2000  # Ultra smooth (slower field gen)
```

### Higher Grid Resolution = Sharper Simulation
```python
GRID_SIZE = 4096  # Default (16.7M cells)
# This is already optimal for 4K!
```

---

## Presets for Different Scenarios

### Testing / Debugging
```python
# Quick Test preset (uncomment in config.py)
GRID_SIZE = 512
NUM_AGENTS = 100_000
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
```

### Maximum Visual Quality
```python
# 4K Ultra preset (uncomment in config.py)
GRID_SIZE = 4096
NUM_AGENTS = 20_000_000
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160
AGENT_SIZE = 0.5
FIELD_SAMPLES = 2000
```

### Stress Test Your GPU
```python
# Extreme Scale preset (uncomment in config.py)
GRID_SIZE = 4096
NUM_AGENTS = 50_000_000
WINDOW_WIDTH = 3840
WINDOW_HEIGHT = 2160
AGENT_SIZE = 0.5
```

---

## 4K-Specific Tips

### 1. Fullscreen Looks Amazing
```python
FULLSCREEN = True  # Immersive experience
```

### 2. Adjust Agent Size to Preference
```python
AGENT_SIZE = 0.7  # Bigger agents (easier to see)
AGENT_SIZE = 0.5  # Smaller agents (more detail, prettier)
```

### 3. Color Modes Look Great at 4K
The default `velocity` mode creates beautiful rainbow patterns that really pop on 4K displays.

### 4. VSync for Smooth Visuals
```python
VSYNC = True  # Recommended (prevents tearing)
```

---

## Troubleshooting 4K Issues

### Window Too Large?
If 4K fills your whole screen uncomfortably:
```python
# Use 1440p instead
WINDOW_WIDTH = 2560
WINDOW_HEIGHT = 1440
GRID_SIZE = 2048
AGENT_SIZE = 0.7
```

### Performance Lower Than Expected?
```python
# Reduce agent count
NUM_AGENTS = 5_000_000  # 5M instead of 10M

# Or reduce grid size
GRID_SIZE = 2048  # Still looks great
```

### Agents Too Small?
```python
AGENT_SIZE = 0.8  # Increase from default 0.6
```

---

## Comparison: 1080p vs 4K

| Setting | 1080p | 4K | Difference |
|---------|-------|-----|------------|
| Resolution | 1920x1080 | 3840x2160 | **4x pixels** |
| Recommended Grid | 2048x2048 | 4096x4096 | **4x cells** |
| Agent Size | 0.8 | 0.6 | Smaller for detail |
| Field Samples | 500 | 1000 | Smoother patterns |
| Visual Quality | Great | **Stunning** | More detail |

---

## Why These Settings?

### 4096x4096 Grid
- **Perfect match** for 4K resolution
- Each pixel can map cleanly to grid cells
- Prevents aliasing and pixelation
- Crisp, clean edges

### 10M Agents
- **Sweet spot** for 4K displays
- Dense enough to look good
- Fast enough to maintain 60 FPS
- Leaves room to scale up

### 0.6 Agent Size
- **Better visibility** of individual agents
- More "breathing room" between agents
- Cleaner visual appearance
- Shows patterns more clearly

### 1000 Field Samples
- **Smoother vector fields**
- More detailed patterns
- Better visual quality at 4K
- Still fast on RTX 4090

---

## Monitor Recommendations

### 4K UHD (3840x2160) ✅
Perfect! Default settings are optimized for you.

### 4K UltraWide (5120x2160 or 3440x1440)
Adjust aspect ratio:
```python
# For 21:9 ultrawide
WINDOW_WIDTH = 3440
WINDOW_HEIGHT = 1440
GRID_SIZE = 3440  # Match width for best results
```

### 1440p (2560x1440)
Still great! Use this preset:
```python
WINDOW_WIDTH = 2560
WINDOW_HEIGHT = 1440
GRID_SIZE = 2560
AGENT_SIZE = 0.7
FIELD_SAMPLES = 750
```

---

## Performance Monitoring

Watch the window title for FPS:
```
Snail Trails GPU - 10,000,000 Agents | FPS: 62.3
```

### What to Expect:
- **60+ FPS** - Perfect! ✅
- **45-60 FPS** - Still smooth
- **30-45 FPS** - Playable, consider reducing agents
- **< 30 FPS** - Reduce NUM_AGENTS or GRID_SIZE

---

## Advanced: Custom Resolutions

### For Any Resolution:
```python
# Match grid size to width for best results
WINDOW_WIDTH = your_width
WINDOW_HEIGHT = your_height
GRID_SIZE = your_width  # Or closest power of 2

# Adjust agent size based on resolution
# Higher res = smaller agents
AGENT_SIZE = 0.8 - (your_width / 10000)  # Rule of thumb
```

---

## Enjoy the Show! 🎮

Your RTX 4090 with a 4K display is the **perfect setup** for this simulation. You'll see incredible detail and smooth performance with 10 million agents!

**Pro tip:** Try fullscreen mode in a dark room for the best experience! 🌌

---

## Quick Reference

```bash
# Run simulation
python snail_trails_modular.py

# Enable fullscreen (edit config.py first)
FULLSCREEN = True

# Test different scales
# Edit NUM_AGENTS in config.py:
# 5M, 10M, 20M, 50M agents

# Adjust visual detail
# Edit AGENT_SIZE in config.py:
# 0.4 (tiny), 0.6 (default), 0.8 (big)
```
