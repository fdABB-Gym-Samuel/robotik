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

### Option B: nix (recommended for the G1 demo)
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
    run_g1_rps_demo.py
    train_rl.py
  src/
    g1_rps/
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
- `run_g1_rps_demo.py` downloads the official `unitree_ros` G1 description into `runs/assets/unitree_ros` on first launch.
- The G1 demo uses Unitree's official `g1_29dof_with_hand_rev_1_0.xml` model.
- On first launch, the project generates a runtime MJCF that inlines the official mesh geometry so MuJoCo can render the real robot appearance inside the Nix shell.

## G1 Hand Demo

The G1 demo now runs entirely through Nix. It currently presents the official
5-finger Unitree Inspire hand as a hand-only MuJoCo scene, which is a better
match for the project milestone when the hand gesture is the main deliverable.

The runtime model is generated from Unitree's official Inspire hand URDF and
its official mesh geometry, so the demo uses the real hand shape rather than
the earlier simplified hand fallback.

Enter the shell and run either the script directly or the dedicated command:

```bash
nix develop
python scripts/run_g1_rps_demo.py
g1-rps-demo
```

Or without entering an interactive shell:

```bash
nix run .#g1-rps-demo
```

This keeps the whole workflow inside WSL + Nix:

- `flake.nix` provides the Python interpreter and Python packages
- the demo runs with the same pinned package set every time
- no virtualenv bootstrap step is needed

Useful options:

```bash
python scripts/run_g1_rps_demo.py --side right
python scripts/run_g1_rps_demo.py --sequence rock scissors paper
python scripts/run_g1_rps_demo.py --hold-seconds 1.5 --transition-seconds 0.8
python scripts/run_g1_rps_demo.py --camera-preset upper_body
```

What the script does:

- Downloads the official Unitree G1 description assets if they are not present yet
- Builds a runtime MuJoCo model from the official 5-finger Inspire hand assets
- Loads the exact hand geometry in a hand-only scene
- Cycles through `rock`, `paper`, and `scissors` with smooth interpolation

## Real Robot Hand Control

The same gesture library now has a real-robot path for the Inspire hand on G1.
The hardware runner publishes to Unitree's official Inspire topics:

- command: `rt/inspire/cmd`
- state: `rt/inspire/state`

The script is dry-run by default, so it prints the outgoing hand channels without
moving the robot unless you explicitly pass `--live`.

Inside the Nix shell:

```bash
nix develop
python scripts/run_g1_rps_hand_hardware.py
python scripts/run_g1_rps_hand_hardware.py --live --interface eth0
```

Or through the Nix app:

```bash
nix run .#g1-rps-hand-hardware -- --live --interface eth0
```

Useful options:

```bash
python scripts/run_g1_rps_hand_hardware.py --sequence rock scissors paper
python scripts/run_g1_rps_hand_hardware.py --hand left
python scripts/run_g1_rps_hand_hardware.py --print-state
python scripts/run_g1_rps_hand_hardware.py --live --interface eth0 --hold-seconds 1.5
```

How the hardware mapping works:

- The MuJoCo hand pose library is more detailed than the real Inspire command API
- Unitree's hand service exposes 6 normalized channels per hand:
  `pinky`, `ring`, `middle`, `index`, `thumb_bend`, `thumb_rotation`
- This project collapses the simulated joints down to those 6 channels and sends them as a 12-slot DDS message, which matches Unitree's official controller layout for both hands

Before using `--live`, make sure:

- the robot-side Inspire hand service is available
- your WSL machine can reach the robot over the chosen network interface
- you test the same sequence in dry-run first
- the hand has enough clearance to open and close safely

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
