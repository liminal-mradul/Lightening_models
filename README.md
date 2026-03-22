# Lightening_models

> **Swarm LED Visualisation** – a simulated 3-D swarm of wearable devices that
> forms coordinated light patterns using **only relative positioning** (no GPS,
> no absolute coordinates).

---

## Overview

Each node in the swarm represents a wearable device equipped with an LED and
local communication capabilities.  Nodes measure distances to nearby neighbours
and collectively infer a local topology through *Classical Multidimensional
Scaling* (MDS).  A per-node **Kalman filter** smooths estimates over time.
Pattern-assignment logic maps visual patterns onto the relative coordinate space
so the whole swarm can display waves, pulsating spheres, checkerboards, radial
rings, and chase effects.

```
swarm/
├── node.py           # Wearable-device node (LED, neighbour detection)
├── environment.py    # 3-D arena with stratified random node placement
├── positioning.py    # Relative positioning: MDS + Kalman filter
├── patterns.py       # Pattern engine (WAVE, SPHERE, CHECKERBOARD, RADIAL, CHASE)
├── simulation.py     # Timestep-based simulation orchestrator
└── visualization.py  # 3-D matplotlib renderer
main.py               # Demo entry point
tests/                # Unit tests (pytest)
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run a 60-step demo cycling through all patterns and save a PNG snapshot
python main.py

# Show a specific pattern for 100 steps
python main.py --steps 100 --pattern wave

# Enable random node movement
python main.py --steps 80 --movement

# Save snapshot to a custom path
python main.py --save my_snapshot.png
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | 60 | Number of simulation timesteps |
| `--nodes N` | 80 | Number of swarm nodes |
| `--pattern NAME` | all | `wave`, `sphere`, `checkerboard`, `radial`, `chase`, or `all` |
| `--save PATH` | `swarm_snapshot.png` | Output PNG path |
| `--seed N` | 42 | Random seed for reproducibility |
| `--movement` | off | Enable random per-step node movement |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## System Design

### Relative Positioning

1. Every node broadcasts a ranging signal; neighbours within the
   *detection radius* reply with a distance measurement.
2. The full pairwise distance matrix is completed via Floyd-Warshall for
   disconnected pairs, then fed to *Classical MDS* to produce a
   globally-consistent (but arbitrary-frame) 3-D embedding.
3. A **constant-velocity Kalman filter** per node smooths the MDS output
   across timesteps, suppressing measurement noise.

### Pattern Mapping

Patterns are defined in a normalised `[-1, 1]³` space and evaluated at each
node's estimated position.  Five built-in patterns:

| Pattern | Effect |
|---------|--------|
| `WAVE` | Sinusoidal colour wave along the X axis |
| `SPHERE` | Pulsating spherical shell |
| `CHECKERBOARD` | 3-D alternating colour cubes with rotating hue |
| `RADIAL` | Concentric rings expanding from centre |
| `CHASE` | Bright band sweeping through the swarm along Z |
