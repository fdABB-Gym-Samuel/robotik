# Project Overview — Robotics Humanoid Course

This document explains what every part of this project does, what the key technologies are, and how everything fits together.

______________________________________________________________________

## What Is This Project?

An 8-week hands-on course where high school students learn to control simulated robots using Python. The learning path moves from a simple robotic arm to a full humanoid:

1. **Reacher** — a 2-joint arm that reaches toward a target
1. **Hopper** — a one-legged robot that must balance and hop forward
1. **Humanoid** — a full-body robot with dozens of joints that walks

Students start by running simulations with random actions, then learn classical control, and finally train AI agents using reinforcement learning.

______________________________________________________________________

## Key Technologies Explained

### MuJoCo (Multi-Joint dynamics with Contact)

A **physics engine** that simulates how rigid bodies, joints, and muscles move and interact. It calculates forces, collisions, and gravity so that simulated robots behave realistically.

Think of it as the "game engine" for robots — it handles all the physics so you can focus on writing the brain (controller) that tells the robot what to do.

- Developed originally at the University of Washington, now maintained by Google DeepMind
- Used by researchers worldwide for robotics and reinforcement learning
- Handles complex contact physics (feet on ground, fingers on objects)

### Gymnasium (formerly OpenAI Gym)

A **Python library** that provides a standard interface for interacting with simulation environments. It wraps MuJoCo environments into a simple loop:

```
observation, reward, done = env.step(action)
```

- **Observation**: what the robot currently sees/senses (joint angles, velocities, positions)
- **Action**: the command you send to the robot (torques applied to each joint)
- **Reward**: a number telling you how well the robot is doing (higher = better)
- **Done**: whether the episode has ended (e.g., the robot fell over)

Gymnasium provides ready-made environments like `Reacher-v5`, `Hopper-v5`, and `Humanoid-v5` so you don't need to build robots from scratch.

### Stable-Baselines3 (SB3)

A **reinforcement learning library** that implements proven RL algorithms. This project uses **PPO (Proximal Policy Optimization)**, which trains a neural network to choose good actions based on observations.

Instead of writing control rules by hand, PPO learns by trial and error:

1. The robot tries random-ish actions
1. It receives rewards for good behavior (moving forward, staying upright)
1. The neural network gradually improves its action choices
1. After thousands of episodes, the robot learns to walk, hop, or reach

### PyTorch

The **deep learning framework** that Stable-Baselines3 uses under the hood. It handles the neural network math — forward passes, backpropagation, and gradient updates. You don't interact with PyTorch directly in this course, but it powers the RL training.

### NumPy

A **numerical computing library** for Python. Used throughout for handling arrays of numbers — observations, actions, and rewards are all NumPy arrays.

### Matplotlib

A **plotting library** for creating charts and graphs. Useful for visualizing training progress, reward curves, and comparing controller performance.

______________________________________________________________________

## Project Structure — What Each Part Does

### Scripts (`scripts/`)

These are the files you run directly.

| Script | What it does |
|--------|-------------|
| `run_reacher.py` | Opens a visual window showing a 2-joint arm taking random actions for 300 steps. Your first simulation. |
| `run_hopper.py` | Runs a one-legged robot with random actions for 500 steps (no visual window). Shows how quickly random control fails at balancing. |
| `run_humanoid.py` | Runs a full humanoid with random actions for 500 steps (no visual window). Demonstrates the complexity of high-dimensional control. |
| `train_rl.py` | Trains a PPO agent on any environment. Saves the trained model and reports how well it performs. This is where the AI learning happens. |

**Usage examples:**

```bash
python scripts/run_reacher.py                           # Watch a random arm
python scripts/train_rl.py --env Hopper-v5 --timesteps 50000  # Train a hopper AI
```

### Controllers (`controllers/`)

Two ways to control a robot, from simplest to classical:

| Controller | How it works |
|-----------|-------------|
| `random_controller.py` | Picks completely random actions. The robot flails around. Used as a baseline to show that control is hard. |
| `pd_controller.py` | Uses **PD control** (Proportional-Derivative) — a classical engineering method. It calculates: `action = Kp × (target - current) + Kd × (-velocity)`. This means: push toward the target (proportional) and dampen oscillations (derivative). Students can tune `Kp` and `Kd` to see how gains affect behavior. |

### Tasks (`tasks/`)

Each subfolder contains a `task.py` with the task name and learning objective:

- **`reacher_task/`** — *"Move the fingertip close to a target using observation-action feedback."*
- **`hopper_task/`** — *"Keep balance while producing forward movement."*
- **`humanoid_task/`** — *"Learn stable full-body movement in simulation."*

These serve as reference cards for what students should focus on at each stage.

### Configs (`configs/`)

| File | Contents |
|------|----------|
| `course_config.yaml` | Course settings: 8-week duration, environment progression (Reacher → Hopper → Humanoid), default training parameters, and output directories. |
| `team_config.yaml` | Classroom info: school year, teacher name, student teams, hardware target (Unitree G1 for future integration). |

### Docs (`docs/`)

| Document | Purpose |
|----------|---------|
| `course_plan.md` | Week-by-week curriculum. Week 1: install and run first sim. Week 2-3: explore Reacher and Hopper. Week 4-5: Ant and Humanoid. Week 6-7: team projects and RL training. Week 8: final demos. |
| `teacher_guide.md` | How to run each class session (mini-lecture → coding → discussion), assessment ideas (reflection journals, checkpoint demos, final presentations), and tips for keeping students engaged. |
| `robot_safety.md` | Safety rules for both simulation and future hardware work. Core principle: *"Simulation first, hardware later."* Covers emergency stops, safe testing zones, action limits, and hardware transition checklists. |

### Output Directories (`runs/`)

| Folder | What goes here |
|--------|---------------|
| `runs/logs/` | Trained model files (`.zip`) and training logs from `train_rl.py` |
| `runs/videos/` | Recorded simulation videos (when enabled) |

______________________________________________________________________

## Core Concepts

### The Reinforcement Learning Loop

```
   ┌──────────┐    action     ┌─────────────┐
   │          │ ──────────→  │             │
   │  Agent   │              │ Environment │
   │ (Policy) │  ←────────── │  (MuJoCo)   │
   │          │  observation  │             │
   └──────────┘  + reward     └─────────────┘
```

1. The **environment** (MuJoCo simulation) provides an **observation** (what the robot senses)
1. The **agent** (neural network or controller) chooses an **action** (joint torques)
1. The environment simulates one time step and returns:
   - A new **observation**
   - A **reward** (how good that action was)
   - Whether the episode is **done** (robot fell, time limit reached)
1. Repeat until the episode ends, then reset and start again

### What the Robots Look Like

- **Reacher**: A flat 2D arm with 2 joints. The goal is a red dot it must reach. Simple, visual, fast.
- **Hopper**: A 2D stick figure with a torso, thigh, leg, and foot. It must hop forward without falling. 4 action dimensions.
- **Humanoid**: A 3D body with torso, two arms, two legs. It must walk forward and stay upright. 17 action dimensions — much harder.

### Why Progressive Complexity?

| Environment | Action dimensions | Observation dimensions | Difficulty |
|------------|-------------------|----------------------|------------|
| Reacher-v5 | 2 | 11 | Low |
| Hopper-v5 | 3 | 11 | Medium |
| Humanoid-v5 | 17 | 376 | High |

Starting simple lets students build intuition about how control works before tackling the complexity of a full humanoid.

______________________________________________________________________

## Future Direction

The course is designed to eventually bridge from simulation to real hardware:

- **Ark Framework** — a robotics orchestration framework for connecting simulation policies to real robots
- **Unitree G1** — a humanoid robot platform. The long-term goal is to transfer learned policies from MuJoCo simulation to a physical G1 robot

This is covered conceptually in Week 7 of the course plan.

______________________________________________________________________

## Dependencies Summary

| Package | Version | Role |
|---------|---------|------|
| `gymnasium[mujoco]` | ≥ 1.0.0 | Simulation environments + MuJoCo physics |
| `stable-baselines3` | ≥ 2.3.0 | PPO reinforcement learning |
| `torch` | ≥ 2.0 | Neural network backend for SB3 |
| `numpy` | ≥ 1.24 | Array math |
| `matplotlib` | ≥ 3.7 | Plotting and visualization |
| Python | 3.10+ | Required language version |
