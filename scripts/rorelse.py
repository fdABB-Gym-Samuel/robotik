"""Run a Unitree G1 rock-paper-scissors demo with synchronized speech."""

from __future__ import annotations

import math
import random
import shutil
import subprocess
import sys
import threading
import traceback
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for local setup
    raise SystemExit(
        "MuJoCo Python bindings are not installed. Run `pip install -r requirements.txt` "
        "before starting the Unitree G1 viewer."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "assets" / "unitree_g1" / "g1_29dof_with_hand.xml"
LOG_PATH = PROJECT_ROOT / "runs" / "logs" / "rorelse-error.log"
FRAME_DT = 1.0 / 60.0
GESTURE_OPTIONS = ("rock", "paper", "scissors")


@dataclass(frozen=True)
class TimingConfig:
    prepare_duration: float = 0.8
    beat_duration: float = 0.5
    speech_offset: float = 0.0
    anticipation_duration: float = 0.08
    reveal_duration: float = 0.18
    hold_duration: float = 1.7
    return_duration: float = 1.0
    idle_duration: float = 0.5


@dataclass(frozen=True)
class MotionConfig:
    base_height: float = 0.79
    arm_amplitude: float = 0.18
    wrist_angle: float = -0.18
    finger_open_close_gain: float = 1.0


@dataclass(frozen=True)
class SpeechConfig:
    voice: str = "Samantha"
    rate: int = 185
    startup_latency: float = 0.12
    estimated_word_duration: float = 0.22


@dataclass(frozen=True)
class DemoConfig:
    scene_path: Path = SCENE_PATH
    frame_dt: float = FRAME_DT
    timing: TimingConfig = TimingConfig()
    motion: MotionConfig = MotionConfig()
    speech: SpeechConfig = SpeechConfig()


@dataclass(frozen=True)
class SpeechCue:
    time_s: float
    text: str


@dataclass
class DemoState:
    model: mujoco.MjModel
    data: mujoco.MjData
    config: DemoConfig
    symbol: str
    idle_pose: dict[str, float]
    ready_pose: dict[str, float]
    reveal_pose: dict[str, float]
    speech_playback: "SpeechPlayback | None" = None
    cycle_started_at: float = 0.0


class SpeechBackend:
    def speak_async(self, text: str) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def close(self) -> None:
        """Release backend resources."""


class MacSayBackend(SpeechBackend):
    def __init__(self, config: SpeechConfig) -> None:
        self._config = config
        self._processes: list[subprocess.Popen[bytes]] = []

    def speak_async(self, text: str) -> None:
        process = subprocess.Popen(
            [
                "say",
                "-v",
                self._config.voice,
                "-r",
                str(self._config.rate),
                text,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._processes.append(process)
        self._processes = [proc for proc in self._processes if proc.poll() is None]

    def close(self) -> None:
        for process in self._processes:
            if process.poll() is None:
                process.terminate()
        self._processes.clear()


class PrintSpeechBackend(SpeechBackend):
    def speak_async(self, text: str) -> None:
        print(f"[speech] {text}")


@dataclass
class SpeechPlayback:
    backend: SpeechBackend
    phrase: str
    anchor_delay: float
    started_at: float | None = None
    anchor_time: float | None = None

    def start(self, now: float) -> None:
        if self.started_at is not None:
            return
        self.backend.speak_async(self.phrase)
        self.started_at = now
        self.anchor_time = now + self.anchor_delay

    def elapsed(self, now: float) -> float | None:
        if self.anchor_time is None:
            return None
        return now - self.anchor_time

    def reset(self) -> None:
        self.started_at = None
        self.anchor_time = None

    def close(self) -> None:
        self.backend.close()


def choose_speech_backend(config: SpeechConfig) -> SpeechBackend:
    if shutil.which("say"):
        return MacSayBackend(config)
    return PrintSpeechBackend()


def ease_in_out_cubic(alpha: float) -> float:
    alpha = clamp_unit(alpha)
    if alpha < 0.5:
        return 4.0 * alpha * alpha * alpha
    return 1.0 - pow(-2.0 * alpha + 2.0, 3.0) / 2.0


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def blend_pose(start: dict[str, float], target: dict[str, float], alpha: float) -> dict[str, float]:
    alpha = clamp_unit(alpha)
    joint_names = set(start) | set(target)
    return {
        joint_name: (1.0 - alpha) * start.get(joint_name, 0.0) + alpha * target.get(joint_name, 0.0)
        for joint_name in joint_names
    }


def add_pose_layers(*layers: dict[str, float]) -> dict[str, float]:
    pose: dict[str, float] = {}
    for layer in layers:
        pose.update(layer)
    return pose


def add_joint_deltas(base_pose: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
    updated_pose = dict(base_pose)
    for joint_name, delta in deltas.items():
        updated_pose[joint_name] = updated_pose.get(joint_name, 0.0) + delta
    return updated_pose


def set_joint_position(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint '{joint_name}' was not found in the Unitree G1 model.")

    if model.jnt_limited[joint_id]:
        low, high = model.jnt_range[joint_id]
        value = max(float(low), min(float(high), value))

    data.qpos[model.jnt_qposadr[joint_id]] = value


def stable_lower_body_pose() -> dict[str, float]:
    return {
        "left_hip_pitch_joint": -0.32,
        "left_knee_joint": 0.64,
        "left_ankle_pitch_joint": -0.33,
        "right_hip_pitch_joint": -0.32,
        "right_knee_joint": 0.64,
        "right_ankle_pitch_joint": -0.33,
    }


def relaxed_left_arm_pose() -> dict[str, float]:
    return {
        "left_shoulder_pitch_joint": 0.22,
        "left_shoulder_roll_joint": 0.1,
        "left_shoulder_yaw_joint": -0.08,
        "left_elbow_joint": 0.48,
        "left_wrist_roll_joint": 0.02,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
    }


def neutral_right_arm_pose() -> dict[str, float]:
    return {
        "right_shoulder_pitch_joint": 0.24,
        "right_shoulder_roll_joint": -0.12,
        "right_shoulder_yaw_joint": 0.02,
        "right_elbow_joint": 0.55,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }


def relaxed_right_hand_pose() -> dict[str, float]:
    return {
        "right_hand_thumb_0_joint": 0.18,
        "right_hand_thumb_1_joint": -0.1,
        "right_hand_thumb_2_joint": -0.18,
        "right_hand_index_0_joint": 0.2,
        "right_hand_index_1_joint": 0.25,
        "right_hand_middle_0_joint": 0.22,
        "right_hand_middle_1_joint": 0.28,
    }


def play_ready_arm_pose(config: MotionConfig) -> dict[str, float]:
    return {
        "waist_yaw_joint": -0.08,
        "waist_roll_joint": -0.03,
        "waist_pitch_joint": 0.04,
        "right_shoulder_pitch_joint": -0.42,
        "right_shoulder_roll_joint": -0.62,
        "right_shoulder_yaw_joint": 0.18,
        "right_elbow_joint": 1.18,
        "right_wrist_roll_joint": -0.08,
        "right_wrist_pitch_joint": config.wrist_angle,
        "right_wrist_yaw_joint": -0.16,
    }


def beat_arm_offset(config: MotionConfig, local_time: float, beat_duration: float, beat_index: int) -> dict[str, float]:
    normalized = clamp_unit(local_time / beat_duration)
    downbeat = math.exp(-18.0 * normalized) if normalized > 0.0 else 1.0
    rebound = math.sin(math.pi * normalized)
    amplitude = config.arm_amplitude

    shoulder_pitch = -amplitude * (0.95 * downbeat - 0.2 * rebound)
    elbow = 0.36 * downbeat - 0.08 * rebound
    wrist_pitch = 0.1 * downbeat
    wrist_roll = -0.03 * beat_index

    return {
        "right_shoulder_pitch_joint": shoulder_pitch,
        "right_elbow_joint": elbow,
        "right_wrist_pitch_joint": wrist_pitch,
        "right_wrist_roll_joint": wrist_roll,
    }


def countdown_hand_pose() -> dict[str, float]:
    return {
        "right_hand_thumb_0_joint": 0.26,
        "right_hand_thumb_1_joint": -0.08,
        "right_hand_thumb_2_joint": -0.22,
        "right_hand_index_0_joint": 0.42,
        "right_hand_index_1_joint": 0.45,
        "right_hand_middle_0_joint": 0.45,
        "right_hand_middle_1_joint": 0.48,
    }


def anticipation_hand_pose() -> dict[str, float]:
    return {
        "right_hand_thumb_0_joint": 0.26,
        "right_hand_thumb_1_joint": -0.08,
        "right_hand_thumb_2_joint": -0.22,
        "right_hand_index_0_joint": 0.42,
        "right_hand_index_1_joint": 0.45,
        "right_hand_middle_0_joint": 0.45,
        "right_hand_middle_1_joint": 0.48,
    }


def final_hand_pose(symbol: str, config: MotionConfig) -> dict[str, float]:
    gain = config.finger_open_close_gain
    poses = {
        "rock": {
            "right_hand_thumb_0_joint": 0.82 * gain,
            "right_hand_thumb_1_joint": 0.36 * gain,
            "right_hand_thumb_2_joint": -1.08 * gain,
            "right_hand_index_0_joint": 1.28 * gain,
            "right_hand_index_1_joint": 1.42 * gain,
            "right_hand_middle_0_joint": 1.28 * gain,
            "right_hand_middle_1_joint": 1.42 * gain,
        },
        "paper": {
            "right_hand_thumb_0_joint": 0.12,
            "right_hand_thumb_1_joint": -0.24,
            "right_hand_thumb_2_joint": -0.05,
            "right_hand_index_0_joint": 0.04,
            "right_hand_index_1_joint": 0.02,
            "right_hand_middle_0_joint": 0.04,
            "right_hand_middle_1_joint": 0.02,
        },
        "scissors": {
            "right_hand_thumb_0_joint": 0.7,
            "right_hand_thumb_1_joint": 0.18,
            "right_hand_thumb_2_joint": -0.88,
            "right_hand_index_0_joint": 0.02,
            "right_hand_index_1_joint": 0.02,
            "right_hand_middle_0_joint": 0.02,
            "right_hand_middle_1_joint": 0.02,
        },
    }
    return poses[symbol]


def reveal_arm_pose(symbol: str, config: MotionConfig) -> dict[str, float]:
    base_pose = {
        "waist_yaw_joint": -0.12,
        "waist_roll_joint": -0.03,
        "waist_pitch_joint": 0.03,
        "right_shoulder_pitch_joint": -0.56,
        "right_shoulder_roll_joint": -0.78,
        "right_shoulder_yaw_joint": 0.26,
        "right_elbow_joint": 1.02,
        "right_wrist_roll_joint": -0.06,
        "right_wrist_pitch_joint": config.wrist_angle - 0.02,
        "right_wrist_yaw_joint": -0.22,
    }

    if symbol == "paper":
        base_pose["right_wrist_pitch_joint"] = config.wrist_angle - 0.14
    elif symbol == "scissors":
        base_pose["right_wrist_yaw_joint"] = -0.34
        base_pose["right_wrist_pitch_joint"] = config.wrist_angle - 0.06
    else:
        base_pose["right_wrist_pitch_joint"] = config.wrist_angle + 0.04

    return base_pose


def reveal_accent_arm_pose(symbol: str, config: MotionConfig) -> dict[str, float]:
    accent_pose = dict(reveal_arm_pose(symbol, config))
    accent_pose["right_shoulder_pitch_joint"] -= 0.1
    accent_pose["right_elbow_joint"] += 0.08
    accent_pose["right_wrist_pitch_joint"] -= 0.12
    return accent_pose


def whole_body_stabilization_pose() -> dict[str, float]:
    return add_pose_layers(
        stable_lower_body_pose(),
        relaxed_left_arm_pose(),
    )


def configure_camera(viewer: mujoco.viewer.Handle) -> None:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 148
    viewer.cam.elevation = -18
    viewer.cam.lookat[:] = [0.0, -0.08, 0.82]


def speech_cues(config: DemoConfig) -> list[SpeechCue]:
    beat = config.timing.beat_duration
    return [
        SpeechCue(0.0 * beat, "Rock"),
        SpeechCue(1.0 * beat, "Paper"),
        SpeechCue(2.0 * beat, "Scissors"),
        SpeechCue(3.0 * beat, "Shoot!"),
    ]


def build_countdown_phrase(config: DemoConfig) -> str:
    cues = speech_cues(config)
    pause_ms = max(
        0,
        int((config.timing.beat_duration - config.speech.estimated_word_duration) * 1000),
    )
    chunks: list[str] = []
    for index, cue in enumerate(cues):
        chunks.append(cue.text)
        if index < len(cues) - 1 and pause_ms > 0:
            chunks.append(f"[[slnc {pause_ms}]]")
    return " ".join(chunks)


def build_idle_pose() -> dict[str, float]:
    return add_pose_layers(
        whole_body_stabilization_pose(),
        neutral_right_arm_pose(),
        relaxed_right_hand_pose(),
    )


def build_ready_pose(config: MotionConfig) -> dict[str, float]:
    return add_pose_layers(
        whole_body_stabilization_pose(),
        play_ready_arm_pose(config),
        countdown_hand_pose(),
    )


def build_reveal_pose(symbol: str, config: MotionConfig) -> dict[str, float]:
    return add_pose_layers(
        whole_body_stabilization_pose(),
        reveal_arm_pose(symbol, config),
        final_hand_pose(symbol, config),
    )


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, config: DemoConfig, pose: dict[str, float]) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:7] = [0.0, 0.0, config.motion.base_height, 1.0, 0.0, 0.0, 0.0]

    for joint_name, value in pose.items():
        set_joint_position(model, data, joint_name, value)

    if model.nu:
        data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)


def prepare(state: DemoState, elapsed: float) -> dict[str, float]:
    alpha = ease_in_out_cubic(elapsed / state.config.timing.prepare_duration)
    return blend_pose(state.idle_pose, state.ready_pose, alpha)


def speak_countdown(state: DemoState) -> SpeechPlayback:
    backend = choose_speech_backend(state.config.speech)
    playback = SpeechPlayback(
        backend=backend,
        phrase=build_countdown_phrase(state.config),
        anchor_delay=state.config.speech.startup_latency + state.config.timing.speech_offset,
    )
    state.speech_playback = playback
    return playback


def countdown_motion(state: DemoState, elapsed: float) -> dict[str, float]:
    beat_duration = state.config.timing.beat_duration
    beat_index = min(int(elapsed / beat_duration), 2)
    local_time = elapsed - beat_index * beat_duration

    base_pose = add_joint_deltas(
        state.ready_pose,
        beat_arm_offset(state.config.motion, local_time, beat_duration, beat_index),
    )

    shoot_time = 3.0 * beat_duration
    anticipation_start = max(shoot_time - state.config.timing.anticipation_duration, 0.0)
    if elapsed >= anticipation_start:
        anticipation_alpha = ease_in_out_cubic(
            (elapsed - anticipation_start) / state.config.timing.anticipation_duration
        )
        anticipation_pose = add_pose_layers(
            whole_body_stabilization_pose(),
            add_joint_deltas(
                play_ready_arm_pose(state.config.motion),
                {
                    "right_shoulder_pitch_joint": 0.08,
                    "right_elbow_joint": -0.08,
                    "right_wrist_pitch_joint": 0.06,
                },
            ),
            countdown_hand_pose(),
        )
        base_pose = blend_pose(base_pose, anticipation_pose, anticipation_alpha)

    # Keep the hand loosely engaged during the spoken beats.
    hand_alpha = 0.45 + 0.15 * math.sin(2.0 * math.pi * clamp_unit(local_time / beat_duration))
    hand_pose = blend_pose(relaxed_right_hand_pose(), countdown_hand_pose(), hand_alpha)
    base_pose.update(hand_pose)
    return base_pose


def reveal(state: DemoState, symbol: str, elapsed: float) -> dict[str, float]:
    anticipation_alpha = clamp_unit(elapsed / state.config.timing.anticipation_duration)
    anticipation_pose = add_pose_layers(
        whole_body_stabilization_pose(),
        reveal_accent_arm_pose(symbol, state.config.motion),
        anticipation_hand_pose(),
    )

    if elapsed <= state.config.timing.anticipation_duration:
        return blend_pose(state.ready_pose, anticipation_pose, ease_in_out_cubic(anticipation_alpha))

    reveal_elapsed = elapsed - state.config.timing.anticipation_duration
    reveal_alpha = ease_in_out_cubic(reveal_elapsed / state.config.timing.reveal_duration)
    return blend_pose(anticipation_pose, state.reveal_pose, reveal_alpha)


def hold(state: DemoState, _elapsed: float) -> dict[str, float]:
    return dict(state.reveal_pose)


def return_to_idle(state: DemoState, elapsed: float) -> dict[str, float]:
    alpha = ease_in_out_cubic(elapsed / state.config.timing.return_duration)
    return blend_pose(state.reveal_pose, state.idle_pose, alpha)


def begin_new_cycle(state: DemoState, now: float) -> None:
    state.symbol = random.choice(GESTURE_OPTIONS)
    state.reveal_pose = build_reveal_pose(state.symbol, state.config.motion)
    state.cycle_started_at = now
    if state.speech_playback is not None:
        state.speech_playback.reset()
    print(f"Next reveal: {state.symbol}")


def build_state(config: DemoConfig) -> DemoState:
    if not config.scene_path.exists():
        raise SystemExit(f"Unitree G1 scene not found: {config.scene_path}")

    model = mujoco.MjModel.from_xml_path(str(config.scene_path))
    data = mujoco.MjData(model)
    initial_symbol = random.choice(GESTURE_OPTIONS)
    state = DemoState(
        model=model,
        data=data,
        config=config,
        symbol=initial_symbol,
        idle_pose=build_idle_pose(),
        ready_pose=build_ready_pose(config.motion),
        reveal_pose=build_reveal_pose(initial_symbol, config.motion),
    )
    return state


def phase_pose(state: DemoState, now: float, elapsed: float) -> dict[str, float]:
    timing = state.config.timing
    prepare_end = timing.prepare_duration
    shoot_start = 3.0 * timing.beat_duration
    reveal_end = shoot_start + timing.anticipation_duration + timing.reveal_duration
    hold_end = reveal_end + timing.hold_duration
    return_end = hold_end + timing.return_duration

    if elapsed < prepare_end:
        return prepare(state, elapsed)

    if state.speech_playback is not None and state.speech_playback.started_at is None:
        state.speech_playback.start(now)

    speech_elapsed = state.speech_playback.elapsed(now) if state.speech_playback is not None else None
    if speech_elapsed is None or speech_elapsed < 0.0:
        return dict(state.ready_pose)

    if speech_elapsed < shoot_start:
        return countdown_motion(state, speech_elapsed)
    if speech_elapsed < reveal_end:
        reveal_elapsed = speech_elapsed - shoot_start
        return reveal(state, state.symbol, reveal_elapsed)
    if speech_elapsed < hold_end:
        return hold(state, speech_elapsed - reveal_end)
    if speech_elapsed < return_end:
        return return_to_idle(state, speech_elapsed - hold_end)
    return dict(state.idle_pose)


def cycle_duration(config: DemoConfig) -> float:
    timing = config.timing
    return (
        timing.prepare_duration
        + max(config.speech.startup_latency + timing.speech_offset, 0.0)
        + 3.0 * timing.beat_duration
        + timing.anticipation_duration
        + timing.reveal_duration
        + timing.hold_duration
        + timing.return_duration
        + timing.idle_duration
    )


def animation_loop(state: DemoState, stop_event: threading.Event) -> None:
    begin_new_cycle(state, now=time.perf_counter())

    while not stop_event.is_set():
        now = time.perf_counter()
        elapsed = now - state.cycle_started_at
        current_cycle_duration = cycle_duration(state.config)

        if elapsed >= current_cycle_duration:
            begin_new_cycle(state, now)
            elapsed = 0.0

        pose = phase_pose(state, now, elapsed)
        apply_pose(state.model, state.data, state.config, pose)
        time.sleep(state.config.frame_dt)


def launch_blocking_fallback(state: DemoState, playback: SpeechPlayback) -> None:
    print("Passive viewer failed. Falling back to blocking viewer.")
    stop_event = threading.Event()
    thread = threading.Thread(target=animation_loop, args=(state, stop_event), daemon=True)
    thread.start()
    try:
        mujoco.viewer._launch_internal(  # type: ignore[attr-defined]
            state.model,
            state.data,
            run_physics_thread=False,
            show_left_ui=True,
            show_right_ui=True,
        )
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        playback.close()


def is_macos_mjpython_requirement(error: Exception) -> bool:
    return sys.platform == "darwin" and "run under `mjpython` on macOS" in str(error)


def main() -> None:
    config = DemoConfig()
    state = build_state(config)
    playback = speak_countdown(state)

    print("Opening Unitree G1 rock-paper-scissors demo...")
    print("Speech: Rock, paper, scissors, shoot!")
    print("Close the MuJoCo window to end the demo.")

    try:
        with mujoco.viewer.launch_passive(state.model, state.data) as viewer:
            configure_camera(viewer)
            begin_new_cycle(state, now=time.perf_counter())

            while viewer.is_running():
                now = time.perf_counter()
                elapsed = now - state.cycle_started_at
                current_cycle_duration = cycle_duration(config)

                if elapsed >= current_cycle_duration:
                    begin_new_cycle(state, now)
                    elapsed = 0.0

                pose = phase_pose(state, now, elapsed)
                apply_pose(state.model, state.data, config, pose)
                viewer.sync()
                time.sleep(config.frame_dt)
    except Exception as error:
        error_trace = traceback.format_exc()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(error_trace)
        if is_macos_mjpython_requirement(error):
            print("Passive viewer is unavailable under plain `python` on macOS.")
            print("For the passive viewer, run: source .venv/bin/activate && mjpython scripts/rorelse.py")
            launch_blocking_fallback(state, playback)
            return
        if sys.platform == "darwin":
            print(f"Passive viewer failed. Details saved to {LOG_PATH}")
            launch_blocking_fallback(state, playback)
            return
        raise

    playback.close()


if __name__ == "__main__":
    main()
