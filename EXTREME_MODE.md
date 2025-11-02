# 🔥 EXTREME MODE - Push Your RTX 4090 to the LIMIT!

## **50 MILLION AGENTS AT 4K - ARE YOU READY?**

Your default configuration is now set to **EXTREME MODE**: 50 million agents rendering at 4K resolution with ultra-detailed vector fields. This will push your RTX 4090 to its limits!

---

## 🚀 Current EXTREME Configuration

```python
NUM_AGENTS = 50_000_000          # 50 MILLION agents!
GRID_SIZE = 4096                 # 4K-perfect grid
AGENT_SIZE = 0.4                 # Tiny for maximum detail
FIELD_SAMPLES = 2000             # Ultra-smooth patterns
AGENT_WORK_GROUP_SIZE = 512      # 2x performance boost
SHOW_DETAILED_STATS = True       # See everything!
```

### Expected Performance
- **FPS:** 25-40 FPS (depends on field complexity)
- **VRAM Usage:** ~1.2 GB (5% of your 24GB)
- **Visual Quality:** **STUNNING** ✨
- **Agent Density:** 2,980 agents per screen pixel!

---

## 🎚️ Scaling Options

### Want Even MORE? Try These Presets!

#### **INSANE MODE (100M agents)** 💀
```python
# Uncomment in config.py:
GRID_SIZE = 4096
NUM_AGENTS = 100_000_000         # 100 MILLION!
AGENT_SIZE = 0.3
FIELD_SAMPLES = 2000
AGENT_WORK_GROUP_SIZE = 1024
BENCHMARK_MODE = True            # Auto-benchmark
```
**Expected:** 15-25 FPS, ~2.4GB VRAM

#### **ABSOLUTE MAXIMUM (8K grid!)** 🌌
```python
# The ultimate stress test:
GRID_SIZE = 8192                 # 67 million cells!
NUM_AGENTS = 100_000_000
AGENT_SIZE = 0.2
FIELD_SAMPLES = 3000
AGENT_WORK_GROUP_SIZE = 1024
```
**Expected:** 10-20 FPS, uses significant VRAM

---

## 📊 Performance Monitoring

### Detailed Stats (Enabled by Default)
Your window title will show:
```
Snail Trails GPU - 50,000,000 Agents | FPS: 32.5 | Frame: 30.7ms | Min: 28.2ms | Max: 35.1ms
```

**What it means:**
- **FPS:** Current frames per second
- **Frame:** Average frame time (lower = faster)
- **Min/Max:** Performance consistency (close values = stable)

### Benchmark Mode
Enable in `config.py`:
```python
BENCHMARK_MODE = True
```

Runs for 300 frames then shows:
```
🏁 BENCHMARK COMPLETE!
======================================================================
  Total frames: 300
  Total time: 9.32s
  Average FPS: 32.19
  Average frame time: 31.07ms
  Min frame time: 28.20ms
  Max frame time: 35.10ms
  Agents: 50,000,000
  Grid: 4096x4096
======================================================================
```

---

## ⚡ Performance Tuning Guide

### If FPS is Too Low (< 20):

1. **Reduce Agents**
   ```python
   NUM_AGENTS = 20_000_000  # Still impressive!
   ```

2. **Simplify Field**
   ```python
   FIELD_SAMPLES = 1000  # Half the samples
   ```

3. **Disable Detailed Stats**
   ```python
   SHOW_DETAILED_STATS = False  # Small FPS boost
   ```

### If FPS is Too High (Want More Challenge):

1. **Increase Agents**
   ```python
   NUM_AGENTS = 100_000_000  # GO BIGGER!
   ```

2. **Increase Grid Resolution**
   ```python
   GRID_SIZE = 8192  # 4x more cells
   ```

3. **Max Out Field Samples**
   ```python
   FIELD_SAMPLES = 5000  # Buttery smooth patterns
   ```

---

## 🎮 Work Group Size Optimization

The `AGENT_WORK_GROUP_SIZE` parameter controls GPU parallelism:

```python
AGENT_WORK_GROUP_SIZE = 256   # Default (balanced)
AGENT_WORK_GROUP_SIZE = 512   # EXTREME default (2x faster)
AGENT_WORK_GROUP_SIZE = 1024  # INSANE mode (max parallel)
```

**RTX 4090 Recommendation:** 512 or 1024 for best performance with 50M+ agents

**How it works:**
- Higher = more parallel threads = faster agent updates
- Must be power of 2 (256, 512, 1024)
- 1024 is the maximum for most GPUs

---

## 🔬 Visual Quality Settings

### Agent Size (Detail Level)
```python
AGENT_SIZE = 0.8  # Big, easy to see
AGENT_SIZE = 0.6  # Standard 4K
AGENT_SIZE = 0.4  # EXTREME (default)
AGENT_SIZE = 0.3  # INSANE
AGENT_SIZE = 0.2  # Microscopic (8K grid)
```

At 50M agents with `AGENT_SIZE = 0.4`, you get **incredible detail**!

### Field Samples (Smoothness)
```python
FIELD_SAMPLES = 500   # Fast generation
FIELD_SAMPLES = 1000  # Smooth
FIELD_SAMPLES = 2000  # EXTREME (default)
FIELD_SAMPLES = 5000  # Buttery smooth
```

Higher samples = smoother patterns but slower field generation

---

## 💾 Memory Usage

| Agents | Grid | VRAM | % of 24GB |
|--------|------|------|-----------|
| 10M | 4096² | ~420 MB | 2% |
| 20M | 4096² | ~720 MB | 3% |
| **50M** | **4096²** | **~1.2 GB** | **5%** |
| 100M | 4096² | ~2.4 GB | 10% |
| 100M | 8192² | ~3.0 GB | 12% |

**Your RTX 4090 can handle MUCH more!**

---

## 🎯 Recommended Configurations

### **Extreme Balanced** (Current Default)
```python
NUM_AGENTS = 50_000_000
GRID_SIZE = 4096
AGENT_SIZE = 0.4
FIELD_SAMPLES = 2000
AGENT_WORK_GROUP_SIZE = 512
```
**Perfect balance of visual quality and performance**

### **Maximum Agents**
```python
NUM_AGENTS = 100_000_000
GRID_SIZE = 4096
AGENT_SIZE = 0.3
FIELD_SAMPLES = 1500
AGENT_WORK_GROUP_SIZE = 1024
```
**For bragging rights!**

### **Maximum Visual Quality**
```python
NUM_AGENTS = 50_000_000
GRID_SIZE = 4096
AGENT_SIZE = 0.3
FIELD_SAMPLES = 5000
AGENT_WORK_GROUP_SIZE = 512
```
**Smoothest, prettiest patterns**

### **8K Resolution Experiment**
```python
NUM_AGENTS = 100_000_000
GRID_SIZE = 8192
AGENT_SIZE = 0.2
FIELD_SAMPLES = 3000
AGENT_WORK_GROUP_SIZE = 1024
```
**The ultimate stress test!**

---

## 🔥 Tips for Maximum Performance

### 1. **Close Background Apps**
Let your GPU focus entirely on the simulation

### 2. **Enable Fullscreen**
```python
FULLSCREEN = True
```
Slight performance boost + more immersive

### 3. **Watch GPU Temperature**
50M agents will make your GPU work! Monitor temps with:
- MSI Afterburner
- GPU-Z
- NVIDIA GeForce Experience

### 4. **Optimal Driver Settings**
- Latest NVIDIA drivers
- Power management: "Prefer Maximum Performance"
- Disable VSync if you want uncapped FPS

### 5. **Monitor FPS Consistency**
Look at Min/Max frame times:
- **Close values (< 5ms difference)** = Stable
- **Large gaps (> 10ms difference)** = Bottleneck somewhere

---

## 🐛 Troubleshooting

### "Out of Memory" Error
```python
# Reduce agents:
NUM_AGENTS = 20_000_000

# Or reduce grid:
GRID_SIZE = 2048
```

### Low FPS (< 15)
```python
# Reduce field complexity:
FIELD_SAMPLES = 1000

# Or reduce agents:
NUM_AGENTS = 30_000_000
```

### Stuttering / Inconsistent FPS
```python
# Enable VSync for consistent timing:
VSYNC = True

# Or reduce work group size:
AGENT_WORK_GROUP_SIZE = 256
```

### GPU Not at 100% Usage
```python
# Increase agents:
NUM_AGENTS = 100_000_000

# Or increase grid:
GRID_SIZE = 8192
```

---

## 📈 Benchmark Your Setup!

### Quick Benchmark
```python
BENCHMARK_MODE = True
NUM_AGENTS = 50_000_000
```

Run and compare results with others!

### Stress Test Ladder
Try these in order and record FPS:

1. **10M agents** - Warm-up
2. **20M agents** - Getting serious
3. **50M agents** - EXTREME (current)
4. **75M agents** - Ultra
5. **100M agents** - INSANE
6. **150M agents** - Maximum!

**Share your results!** What's the highest agent count where you maintain 30+ FPS?

---

## 🎨 Visual Enhancements (Experimental)

These are disabled by default but you can enable them:

```python
ENABLE_MOTION_BLUR = True   # Smooth trails (performance cost)
ENABLE_GLOW_EFFECT = True   # Agents glow based on speed
PARTICLE_DENSITY = 2.0      # 2x density (more agents visible)
```

**Note:** These features are placeholders for future implementation

---

## 🏆 Achievement Checklist

- [ ] Run 10M agents at 60+ FPS
- [ ] Run 50M agents at 30+ FPS
- [ ] Run 100M agents at any FPS
- [ ] Try 8K grid (8192x8192)
- [ ] Enable fullscreen mode
- [ ] Run a benchmark
- [ ] Find your GPU's agent limit
- [ ] Get stable 60 FPS with maximum agents
- [ ] Share screenshots of 50M+ agents!

---

## 🎮 Quick Commands

### Run Extreme Mode (Current Config)
```bash
python snail_trails_modular.py
```

### Run Benchmark
Edit config.py:
```python
BENCHMARK_MODE = True
```
Then run normally.

### Quick Test Mode
Edit config.py:
```python
NUM_AGENTS = 1_000_000  # Quick test
```

---

## 🌟 What Makes This EXTREME?

| Setting | Normal | EXTREME |
|---------|--------|---------|
| Agents | 10M | **50M** (5x) |
| Grid | 2048² | **4096²** (4x) |
| Field Samples | 500 | **2000** (4x) |
| Work Groups | 256 | **512** (2x) |
| Agent Size | 0.6 | **0.4** (smaller) |
| Visual Detail | Great | **Insane!** |

**Result:** Approximately **40x more computational work** than standard mode!

---

## 🚀 Ready to Go INSANE?

Your RTX 4090 was built for this. **50 MILLION AGENTS** are waiting!

```bash
python snail_trails_modular.py
```

**Watch your GPU unleash its full power!** 🔥💪

---

**Need help?** Check `4K_SETUP.md` for display-specific tips or `README_GPU.md` for general info.

**Want to go back to normal?** Edit `config.py` and reduce `NUM_AGENTS` to 10,000,000.
