"""Simple bicep curl demo for the Unitree G1.

Connects to the G1 over DDS, enables the arm_sdk channel, and cycles the
elbow joints between extension (~0 rad) and flexion (~pi/2 rad) a few times.

Usage:
    python curl_biceps.py [network_interface]

Pass the network interface that is on the same subnet as the robot (e.g. eth0).
Leaving it off uses the SDK default.
"""

import math
import sys
import time

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


# G1 joint indices (29 DoF layout). Only the ones we touch are listed.
class G1Joint:
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    # Weight channel: setting .q = 1 hands control to arm_sdk, 0 releases it.
    kNotUsedJoint = 29


# Joints we actively hold. We pin the shoulders near zero so the arms stay
# at the sides and only the elbows move.
HELD_JOINTS = [
    G1Joint.LeftShoulderPitch,
    G1Joint.LeftShoulderRoll,
    G1Joint.LeftShoulderYaw,
    G1Joint.LeftElbow,
    G1Joint.RightShoulderPitch,
    G1Joint.RightShoulderRoll,
    G1Joint.RightShoulderYaw,
    G1Joint.RightElbow,
]

ELBOW_JOINTS = (G1Joint.LeftElbow, G1Joint.RightElbow)

CONTROL_DT = 0.02  # 50 Hz command loop
KP = 60.0
KD = 1.5
EXTEND_Q = 0.0  # arm straight
CURL_Q = math.pi / 2  # elbow flexed ~90 deg
RAMP_IN_S = 2.0  # smooth move into starting pose
HALF_CYCLE_S = 1.5  # time to go from extended -> curled (or back)
NUM_REPS = 5
RAMP_OUT_S = 2.0  # smooth handoff back to the robot


class BicepCurler:
    def __init__(self):
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: LowState_ | None = None
        self.got_state = False
        self.crc = CRC()
        self.t = 0.0
        self.done = False
        self.start_q = {}  # captured at t=0 for smooth ramp-in

    def init(self):
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

    def start(self):
        while not self.got_state:
            print("Waiting for lowstate...")
            time.sleep(0.5)
        # Snapshot joint positions so we can ramp from wherever the arms are now.
        self.start_q = {j: self.low_state.motor_state[j].q for j in HELD_JOINTS}
        self.loop = RecurrentThread(
            interval=CONTROL_DT, target=self._tick, name="bicep_curl"
        )
        self.loop.Start()

    def _on_state(self, msg: LowState_):
        self.low_state = msg
        self.got_state = True

    def _set_joint(self, joint, q):
        mc = self.low_cmd.motor_cmd[joint]
        mc.tau = 0.0
        mc.q = q
        mc.dq = 0.0
        mc.kp = KP
        mc.kd = KD

    def _tick(self):
        self.t += CONTROL_DT

        curl_total = 2 * HALF_CYCLE_S * NUM_REPS
        t_ramp_out_start = RAMP_IN_S + curl_total
        t_end = t_ramp_out_start + RAMP_OUT_S

        # Keep arm_sdk engaged for the whole motion; we fade it out at the end.
        self.low_cmd.motor_cmd[G1Joint.kNotUsedJoint].q = 1.0

        if self.t < RAMP_IN_S:
            # Ramp from captured start pose to shoulders=0, elbows extended.
            ratio = np.clip(self.t / RAMP_IN_S, 0.0, 1.0)
            for j in HELD_JOINTS:
                target = EXTEND_Q if j in ELBOW_JOINTS else 0.0
                q = (1.0 - ratio) * self.start_q[j] + ratio * target
                self._set_joint(j, q)

        elif self.t < t_ramp_out_start:
            # Curl cycles: elbows oscillate between EXTEND_Q and CURL_Q.
            local = self.t - RAMP_IN_S
            phase = (local % (2 * HALF_CYCLE_S)) / HALF_CYCLE_S  # 0..2
            if phase <= 1.0:
                ratio = phase  # extending -> curling
            else:
                ratio = 2.0 - phase  # curling -> extending
            elbow_q = (1.0 - ratio) * EXTEND_Q + ratio * CURL_Q
            for j in HELD_JOINTS:
                if j in ELBOW_JOINTS:
                    self._set_joint(j, elbow_q)
                else:
                    self._set_joint(j, 0.0)

        elif self.t < t_end:
            # Hold pose and smoothly release arm_sdk weight back to the robot.
            ratio = np.clip((self.t - t_ramp_out_start) / RAMP_OUT_S, 0.0, 1.0)
            for j in HELD_JOINTS:
                target = EXTEND_Q if j in ELBOW_JOINTS else 0.0
                self._set_joint(j, target)
            self.low_cmd.motor_cmd[G1Joint.kNotUsedJoint].q = 1.0 - ratio

        else:
            self.done = True
            return

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def main():
    print("WARNING: clear space around the robot before running this.")
    input("Press Enter to start bicep curls...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    curler = BicepCurler()
    curler.init()
    curler.start()

    while not curler.done:
        time.sleep(0.2)
    print("Done.")


if __name__ == "__main__":
    main()
