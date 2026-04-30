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
constexpr float kRightElbow90DegQ = 0.262f;
constexpr float kRightElbow125DegQ = 0.873f;
constexpr std::array<int, 7> kRightArmJoints = {22, 23, 24, 25, 26, 27, 28};
constexpr std::array<const char *, 7> kRightArmNames = {
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
};

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
  double setup_duration = 0.8;
  double beat_duration = 0.5;
  int beat_count = 3;
  double return_duration = 0.8;
  double release_duration = 0.6;
  float kp = 60.0f;
  float kd = 1.5f;
  float hold_kp = 60.0f;
  float hold_kd = 1.5f;
  float arm_amplitude = 0.02f;
  float wrist_angle = -0.18f;
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

std::array<float, 7> ReadyPose(const Config &config) {
  return {
      -0.785f,
      0.0f,
      0.0f,
      kRightElbow90DegQ,
      -0.08f,
      config.wrist_angle,
      -0.16f,
  };
}

std::array<float, 7> BlendPose(
    const std::array<float, 7> &start,
    const std::array<float, 7> &target,
    double alpha) {
  alpha = ClampUnit(alpha);
  std::array<float, 7> pose{};
  for (size_t i = 0; i < pose.size(); ++i) {
    pose[i] = static_cast<float>((1.0 - alpha) * start[i] + alpha * target[i]);
  }
  return pose;
}

std::array<float, 7> PhasePose(
    const Config &config,
    const std::array<float, 7> &ready_pose,
    int beat_index,
    double local_time) {
  std::array<float, 7> pose = ready_pose;
  const double normalized = ClampUnit(local_time / config.beat_duration);
  const double extension = std::sin(M_PI * normalized);

  pose[0] += static_cast<float>(config.arm_amplitude * extension);
  pose[3] += (kRightElbow125DegQ - kRightElbow90DegQ) *
             static_cast<float>(extension);
  pose[5] += static_cast<float>(0.04 * extension);
  pose[4] += static_cast<float>(-0.03 * beat_index);
  return pose;
}

bool IsRightArmJoint(int joint_index) {
  return std::find(kRightArmJoints.begin(), kRightArmJoints.end(), joint_index) !=
         kRightArmJoints.end();
}

void SleepSeconds(double seconds) {
  if (seconds > 0.0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  }
}

void PrintUsage() {
  std::cout
      << "Usage: g1_pre_reveal_right_arm_hardware_cpp --interface IFACE [--live]\n"
      << "       [--domain-id N] [--print-state] [--no-auto-release]\n";
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

  std::array<float, 7> CurrentRightArmPose() {
    const LowState_ state = StateSnapshot();
    std::array<float, 7> pose{};
    for (size_t i = 0; i < kRightArmJoints.size(); ++i) {
      pose[i] = state.motor_state().at(kRightArmJoints[i]).q();
    }
    return pose;
  }

  void PublishPose(const std::array<float, 7> &pose) {
    LowCmd_ low_cmd;
    low_cmd.mode_pr() = 0;
    low_cmd.mode_machine() = ModeMachine();

    for (int joint_index = 0; joint_index < kNumG1Motors; ++joint_index) {
      auto &motor_cmd = low_cmd.motor_cmd().at(joint_index);
      motor_cmd.mode() = 1;
      motor_cmd.tau() = 0.0f;
      motor_cmd.dq() = 0.0f;
      motor_cmd.q() = hold_q_[joint_index];
      if (IsRightArmJoint(joint_index)) {
        motor_cmd.kp() = config_.kp;
        motor_cmd.kd() = config_.kd;
      } else {
        motor_cmd.kp() = config_.hold_kp;
        motor_cmd.kd() = config_.hold_kd;
      }
    }

    for (size_t i = 0; i < kRightArmJoints.size(); ++i) {
      low_cmd.motor_cmd().at(kRightArmJoints[i]).q() = pose[i];
    }

    low_cmd.crc() =
        Crc32Core(reinterpret_cast<uint32_t *>(&low_cmd),
                  (sizeof(LowCmd_) >> 2) - 1);
    lowcmd_publisher_->Write(low_cmd);
  }

  void Interpolate(
      const std::array<float, 7> &start_pose,
      const std::array<float, 7> &target_pose,
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

  void Hold(const std::array<float, 7> &pose, double duration) {
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
    const auto ready_pose = ReadyPose(config);

    if (!config.live && !config.print_state) {
      std::cout << "Dry run only. No real robot commands were sent.\n";
      std::cout << "Planned ready pose:\n";
      for (size_t i = 0; i < kRightArmNames.size(); ++i) {
        std::cout << "  " << kRightArmNames[i] << ": " << ready_pose[i] << "\n";
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
    const auto start_pose = session.CurrentRightArmPose();

    if (config.print_state) {
      std::cout << "Initial right-arm joint state:\n";
      for (size_t i = 0; i < kRightArmNames.size(); ++i) {
        std::cout << "  " << kRightArmNames[i] << ": " << start_pose[i] << "\n";
      }
    }

    if (!config.live) {
      std::cout << "Dry run only. No real robot commands were sent.\n";
      return 0;
    }

    std::cout << "Sending `rt/lowcmd` with the C++ Unitree SDK backend.\n";
    std::cout << "Moving real G1 right arm into the concealed ready pose...\n";
    session.Interpolate(start_pose, ready_pose, config.setup_duration);

    for (int beat_index = 0; beat_index < config.beat_count; ++beat_index) {
      std::cout << "Running pre-reveal beat " << (beat_index + 1) << "/"
                << config.beat_count << "...\n";
      const int steps =
          std::max(1, static_cast<int>(config.beat_duration / config.control_dt));
      for (int step = 0; step < steps; ++step) {
        const double local_time = step * config.control_dt;
        session.PublishPose(PhasePose(config, ready_pose, beat_index, local_time));
        SleepSeconds(config.control_dt);
      }
    }

    std::cout << "Returning the real G1 right arm to the concealed ready pose...\n";
    session.Interpolate(
        PhasePose(config, ready_pose, config.beat_count - 1, config.beat_duration),
        ready_pose,
        config.return_duration);
    std::cout << "Holding ready pose. Re-engage the high-level controller externally "
              << "when finished.\n";
    session.Hold(ready_pose, config.release_duration);
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
