# robotics-humanoid-course

A beginner-friendly robotics education repository for high school students learning simulation, control, and reinforcement learning with Python, MuJoCo, and Gymnasium.

## Project Overview

This repository provides a practical path from simple robot control to humanoid simulation:

- Run MuJoCo environments with readable Python scripts
- Understand rewards, observations, and actions
- Compare random and simple controller behavior
- Train reinforcement learning agents with Stable-Baselines3 (PPO)
- Build toward future humanoid workflows (Ark framework and Unitree G1 integration)

## Course Description

The course is designed for an 8-week classroom format with low setup friction. Students start running simulations in 10-15 minutes and then progress from:

1. Reacher (arm control)
1. Hopper (balance and locomotion)
1. Humanoid (full-body control)

Supporting documents for teachers and classroom safety are under `docs/`.

## Setup Instructions

### Option A: pip (fastest)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Option B: nix (best)
This option requires the [nix package manager](https://nixos.org/download/)
```bash
nix develop
```

### Option C: conda

```bash
conda env create -f environment.yml
conda activate robotics-humanoid-course
```

## Quick Start Guide

```bash
pip install -r requirements.txt # Do not run this if your using the nix package manager
python scripts/run_reacher.py
```

Then try:

```bash
python scripts/run_hopper.py
python scripts/run_humanoid.py
```

Train a reinforcement learning policy:

```bash
python scripts/train_rl.py --env Hopper-v5 --timesteps 50000
```

## Repository Layout

```text
robotics-humanoid-course/
  README.md
  requirements.txt
  environment.yml
  scripts/
    run_reacher.py
    run_hopper.py
    run_humanoid.py
    train_rl.py
  controllers/
    random_controller.py
    pd_controller.py
  tasks/
    reacher_task/
    hopper_task/
    humanoid_task/
  configs/
    course_config.yaml
    team_config.yaml
  docs/
    course_plan.md
    teacher_guide.md
    robot_safety.md
  runs/
    logs/
    videos/
```

## Notes

- Python 3.10+ is recommended.
- MuJoCo is installed automatically via `gymnasium[mujoco]`.
- If rendering issues occur on school computers, use non-render mode first and confirm installation with headless runs.

## References

- Gymnasium: https://gymnasium.farama.org/
- MuJoCo envs: https://gymnasium.farama.org/environments/mujoco/
- Reacher: https://gymnasium.farama.org/environments/mujoco/reacher/
- Hopper: https://gymnasium.farama.org/environments/mujoco/hopper/
- Humanoid: https://gymnasium.farama.org/environments/mujoco/humanoid/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
- MuJoCo docs: https://mujoco.readthedocs.io/en/stable/
- Ark framework: https://github.com/Robotics-Ark/ark_framework
- Unitree GitHub: https://github.com/unitreerobotics
