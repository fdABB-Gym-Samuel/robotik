// Drive the real Unitree G1 into the celebration "winning" pose: both arms
// raised toward the ceiling. The legs and torso are held at whatever pose the
// robot was in when this binary opened the lowstate channel; only the
// fourteen arm joints (left + right shoulder/elbow/wrist) are driven.
//
// The target joint values are kept in sync with
// `g1_rps.arm_hardware.winning_pose` (Python). If you change the pose on one
// side, mirror it on the other.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;
using unitree::robot::b2::MotionSwitcherClient;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

namespace {

constexpr int kNumG1Motors = 29;
constexpr size_t kArmJointCount = 14;

// Left-arm motor indices (15..21) followed by right-arm motor indices (22..28).
constexpr std::array<int, kArmJointCount> kArmJoints = {
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
};
constexpr std::array<const char *, kArmJointCount> kArmNames = {
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
};

// Shoulder pitch ~175 deg, just inside the URDF range [-3.09, 2.67]. Mirror
// of `WINNING_SHOULDER_PITCH` in `g1_rps/arm_hardware.py`.
constexpr float kWinningShoulderPitch = -3.05f;

// Compensation for the 16 deg tilt baked into each shoulder_pitch_link parent
// quat: rotating the arm ~180 deg about that tilted Y axis leaves the arm
// tilted ~32 deg toward the body's midline. +-0.56 rad of shoulder_roll
// rotates each arm back outward to true vertical (positive on the left,
// negative on the right). Mirror of `WINNING_SHOULDER_ROLL` in
// `g1_rps/arm_hardware.py`.
constexpr float kWinningShoulderRoll = 0.56f;

// Slight elbow bend. In this URDF q=0 is already a heavy bend (~75 deg
// interior), so a "nearly straight" elbow needs a positive q -- q=1.7
// corresponds to ~172 deg interior, about 8 deg of flex. Mirror of
// `WINNING_ELBOW` in `g1_rps/arm_hardware.py`.
constexpr float kWinningElbow = 1.7f;

struct Config {
  std::string interface;
  int domain_id = 0;
  bool live = false;
  bool print_state = false;
  bool auto_release_mode = true;
  double state_timeout_seconds = 5.0;
  double motion_switch_timeout_seconds = 5.0;
  double motion_switch_poll_interval = 0.5;
  double control_dt = 0.005;
  double setup_duration = 1.5;
  double hold_duration = 5.0;
  float kp = 60.0f;
  float kd = 1.5f;
  float hold_kp = 60.0f;
  float hold_kd = 1.5f;
};

uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t crc32 = 0xFFFFFFFF;
  const uint32_t polynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (crc32 & 0x80000000) {
        crc32 <<= 1;
        crc32 ^= polynomial;
      } else {
        crc32 <<= 1;
      }
      if (data & xbit) {
        crc32 ^= polynomial;
      }
      xbit >>= 1;
    }
  }
  return crc32;
}

double ClampUnit(double value) {
  return std::max(0.0, std::min(1.0, value));
}

double EaseInOutCubic(double alpha) {
  alpha = ClampUnit(alpha);
  if (alpha < 0.5) {
    return 4.0 * alpha * alpha * alpha;
  }
  return 1.0 - std::pow(-2.0 * alpha + 2.0, 3.0) / 2.0;
}

// Both arms raised toward the ceiling, with a slight elbow bend. Order
// matches `kArmJoints` / `kArmNames`: left arm (7 joints), then right arm
// (7 joints). Indices 1/8 are shoulder_roll, 3/10 are the elbow joints.
std::array<float, kArmJointCount> WinningPose() {
  std::array<float, kArmJointCount> pose{};
  pose.fill(0.0f);
  pose[0] = kWinningShoulderPitch;   // left_shoulder_pitch_joint
  pose[1] = kWinningShoulderRoll;    // left_shoulder_roll_joint
  pose[3] = kWinningElbow;           // left_elbow_joint
  pose[7] = kWinningShoulderPitch;   // right_shoulder_pitch_joint
  pose[8] = -kWinningShoulderRoll;   // right_shoulder_roll_joint
  pose[10] = kWinningElbow;          // right_elbow_joint
  return pose;
}

std::array<float, kArmJointCount> BlendPose(
    const std::array<float, kArmJointCount> &start,
    const std::array<float, kArmJointCount> &target,
    double alpha) {
  alpha = ClampUnit(alpha);
  std::array<float, kArmJointCount> pose{};
  for (size_t i = 0; i < pose.size(); ++i) {
    pose[i] = static_cast<float>((1.0 - alpha) * start[i] + alpha * target[i]);
  }
  return pose;
}

bool IsArmJoint(int joint_index) {
  return std::find(kArmJoints.begin(), kArmJoints.end(), joint_index) !=
         kArmJoints.end();
}

void SleepSeconds(double seconds) {
  if (seconds > 0.0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  }
}

void PrintUsage() {
  std::cout
      << "Usage: g1_winning_pose_hardware_cpp --interface IFACE [--live]\n"
      << "       [--domain-id N] [--print-state] [--no-auto-release]\n"
      << "       [--hold-duration SECONDS]\n";
}

Config ParseArgs(int argc, char **argv) {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      std::exit(0);
    } else if (arg == "--interface") {
      config.interface = require_value(arg);
    } else if (arg == "--domain-id") {
      config.domain_id = std::stoi(require_value(arg));
    } else if (arg == "--live") {
      config.live = true;
    } else if (arg == "--print-state") {
      config.print_state = true;
    } else if (arg == "--no-auto-release") {
      config.auto_release_mode = false;
    } else if (arg == "--state-timeout-seconds") {
      config.state_timeout_seconds = std::stod(require_value(arg));
    } else if (arg == "--motion-switch-timeout-seconds") {
      config.motion_switch_timeout_seconds = std::stod(require_value(arg));
    } else if (arg == "--hold-duration") {
      config.hold_duration = std::stod(require_value(arg));
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (config.interface.empty()) {
    throw std::runtime_error("The C++ backend requires --interface.");
  }
  return config;
}

class UnitreeCppArmSession {
 public:
  explicit UnitreeCppArmSession(const Config &config) : config_(config) {
    ChannelFactory::Instance()->Init(config_.domain_id, config_.interface);
    MaybeReleaseHighLevelMode();

    lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>("rt/lowcmd"));
    lowcmd_publisher_->InitChannel();

    lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>("rt/lowstate"));
    lowstate_subscriber_->InitChannel(
        [this](const void *message) { LowStateHandler(message); },
        1);
  }

  bool WaitForState() {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(
                              config_.state_timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_state_) {
          return true;
        }
      }
      SleepSeconds(0.05);
    }
    return false;
  }

  LowState_ StateSnapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_state_) {
      throw std::runtime_error("No lowstate sample has been received.");
    }
    return low_state_;
  }

  void CaptureHoldPose() {
    const LowState_ state = StateSnapshot();
    for (int i = 0; i < kNumG1Motors; ++i) {
      hold_q_[i] = state.motor_state().at(i).q();
    }
  }

  std::array<float, kArmJointCount> CurrentArmPose() {
    const LowState_ state = StateSnapshot();
    std::array<float, kArmJointCount> pose{};
    for (size_t i = 0; i < kArmJoints.size(); ++i) {
      pose[i] = state.motor_state().at(kArmJoints[i]).q();
    }
    return pose;
  }

  void PublishPose(const std::array<float, kArmJointCount> &pose) {
    LowCmd_ low_cmd;
    low_cmd.mode_pr() = 0;
    low_cmd.mode_machine() = ModeMachine();

    for (int joint_index = 0; joint_index < kNumG1Motors; ++joint_index) {
      auto &motor_cmd = low_cmd.motor_cmd().at(joint_index);
      motor_cmd.mode() = 1;
      motor_cmd.tau() = 0.0f;
      motor_cmd.dq() = 0.0f;
      motor_cmd.q() = hold_q_[joint_index];
      if (IsArmJoint(joint_index)) {
        motor_cmd.kp() = config_.kp;
        motor_cmd.kd() = config_.kd;
      } else {
        motor_cmd.kp() = config_.hold_kp;
        motor_cmd.kd() = config_.hold_kd;
      }
    }

    for (size_t i = 0; i < kArmJoints.size(); ++i) {
      low_cmd.motor_cmd().at(kArmJoints[i]).q() = pose[i];
    }

    low_cmd.crc() =
        Crc32Core(reinterpret_cast<uint32_t *>(&low_cmd),
                  (sizeof(LowCmd_) >> 2) - 1);
    lowcmd_publisher_->Write(low_cmd);
  }

  void Interpolate(
      const std::array<float, kArmJointCount> &start_pose,
      const std::array<float, kArmJointCount> &target_pose,
      double duration) {
    const int steps = std::max(1, static_cast<int>(duration / config_.control_dt));
    for (int step = 1; step <= steps; ++step) {
      const double alpha = EaseInOutCubic(static_cast<double>(step) / steps);
      if (config_.live) {
        PublishPose(BlendPose(start_pose, target_pose, alpha));
      }
      SleepSeconds(config_.control_dt);
    }
  }

  void Hold(const std::array<float, kArmJointCount> &pose, double duration) {
    const int steps = std::max(1, static_cast<int>(duration / config_.control_dt));
    for (int step = 0; step < steps; ++step) {
      if (config_.live) {
        PublishPose(pose);
      }
      SleepSeconds(config_.control_dt);
    }
  }

 private:
  void LowStateHandler(const void *message) {
    const LowState_ state = *static_cast<const LowState_ *>(message);
    const uint32_t expected =
        Crc32Core(reinterpret_cast<uint32_t *>(
                      const_cast<LowState_ *>(&state)),
                  (sizeof(LowState_) >> 2) - 1);
    if (state.crc() != expected) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    low_state_ = state;
    mode_machine_ = state.mode_machine();
    has_state_ = true;
  }

  uint8_t ModeMachine() {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_machine_;
  }

  void MaybeReleaseHighLevelMode() {
    if (!config_.auto_release_mode) {
      return;
    }

    MotionSwitcherClient client;
    client.SetTimeout(static_cast<float>(config_.motion_switch_timeout_seconds));
    client.Init();

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration<double>(
                              config_.motion_switch_timeout_seconds);
    while (std::chrono::steady_clock::now() < deadline) {
      std::string form;
      std::string name;
      const int32_t status = client.CheckMode(form, name);
      if (status != 0) {
        std::cout << "MotionSwitcherClient.CheckMode did not respond. "
                  << "Release manually with L2+B then L2+R2.\n";
        return;
      }
      if (name.empty()) {
        return;
      }
      if (client.ReleaseMode() != 0) {
        std::cout << "MotionSwitcherClient.ReleaseMode did not respond. "
                  << "Release manually with L2+B then L2+R2.\n";
        return;
      }
      SleepSeconds(config_.motion_switch_poll_interval);
    }

    std::cout << "Timed out while asking MotionSwitcherClient to release the "
              << "high-level mode. Continuing anyway.\n";
  }

  Config config_;
  std::mutex mutex_;
  LowState_ low_state_;
  bool has_state_ = false;
  uint8_t mode_machine_ = 0;
  std::array<float, kNumG1Motors> hold_q_{};
  std::shared_ptr<ChannelPublisher<LowCmd_>> lowcmd_publisher_;
  std::shared_ptr<ChannelSubscriber<LowState_>> lowstate_subscriber_;
};

}  // namespace

int main(int argc, char **argv) {
  try {
    const Config config = ParseArgs(argc, argv);
    const auto winning_pose = WinningPose();

    if (!config.live && !config.print_state) {
      std::cout << "Dry run only. No real robot commands were sent.\n";
      std::cout << "Planned winning pose:\n";
      for (size_t i = 0; i < kArmNames.size(); ++i) {
        std::cout << "  " << kArmNames[i] << ": " << winning_pose[i] << "\n";
      }
      return 0;
    }

    UnitreeCppArmSession session(config);
    if (!session.WaitForState()) {
      std::cerr << "No `rt/lowstate` sample was received by the C++ Unitree SDK "
                << "backend on interface " << config.interface << ", domain "
                << config.domain_id << ".\n";
      return 1;
    }

    session.CaptureHoldPose();
    const auto start_pose = session.CurrentArmPose();

    if (config.print_state) {
      std::cout << "Initial arm joint state:\n";
      for (size_t i = 0; i < kArmNames.size(); ++i) {
        std::cout << "  " << kArmNames[i] << ": " << start_pose[i] << "\n";
      }
    }

    if (!config.live) {
      std::cout << "Dry run only. No real robot commands were sent.\n";
      return 0;
    }

    std::cout << "Sending `rt/lowcmd` with the C++ Unitree SDK backend.\n";
    std::cout << "Raising both G1 arms into the winning pose...\n";
    session.Interpolate(start_pose, winning_pose, config.setup_duration);
    std::cout << "Holding winning pose for " << config.hold_duration
              << "s. Re-engage the high-level controller externally afterwards.\n";
    session.Hold(winning_pose, config.hold_duration);
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
