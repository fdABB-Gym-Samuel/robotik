# robotics-humanoid-course

A beginner-friendly robotics education repository for high school students learning simulation, control, and reinforcement learning with Python, MuJoCo, and Gymnasium.

## Project Overview

This repository provides a practical path from simple robot control to humanoid simulation:

- Run MuJoCo environments with readable Python scripts
- Understand rewards, observations, and actions
- Compare random and simple controller behavior
- Train reinforcement learning agents with Stable-Baselines3 (PPO)
- Use a local Unitree G1 MuJoCo scene for humanoid-specific experiments

## Course Description

The course is designed for an 8-week classroom format with low setup friction. Students start running simulations in 10-15 minutes and then progress from:

1. Reacher (arm control)
1. Hopper (balance and locomotion)
1. Unitree G1 (full-body control)

Supporting documents for teachers and classroom safety are under `docs/`.

## Setup Instructions

### Python venv

```bash
chmod +x scripts/setup-venv
./scripts/setup-venv
source .venv/bin/activate
```

## Quick Start Guide

```bash
python scripts/run_reacher.py
```

Then try:

```bash
python scripts/run_hopper.py
python scripts/run_humanoid.py
```

For the Unitree G1 viewer on macOS, use:

```bash
./scripts/run-g1-sim
```

Train a reinforcement learning policy:

```bash
python scripts/train_rl.py --env UnitreeG1-v0 --timesteps 50000
```

## Repository Layout

```text
robotics-humanoid-course/
  README.md
  requirements.txt
  scripts/
    setup-venv
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
  environments/
    unitree_g1_env.py
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
- The humanoid workflow expects a local G1 scene at `assets/unitree_g1/g1_29dof_with_hand.xml`.
- If rendering issues occur on school computers, use non-render mode first and confirm installation with headless runs.

## References

- Gymnasium: https://gymnasium.farama.org/
- MuJoCo envs: https://gymnasium.farama.org/environments/mujoco/
- Reacher: https://gymnasium.farama.org/environments/mujoco/reacher/
- Hopper: https://gymnasium.farama.org/environments/mujoco/hopper/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
- MuJoCo docs: https://mujoco.readthedocs.io/en/stable/
- Ark framework: https://github.com/Robotics-Ark/ark_framework
- Unitree GitHub: https://github.com/unitreerobotics
