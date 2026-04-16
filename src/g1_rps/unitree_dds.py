"""Minimal Unitree DDS bridge for the Inspire hand topics.

This module mirrors the small subset of the official `unitree_sdk2_python`
package that we need for the G1 Inspire hand:

- DDS participant setup with an optional network interface
- `unitree_go.msg.dds_.MotorCmds_` on `rt/inspire/cmd`
- `unitree_go.msg.dds_.MotorStates_` on `rt/inspire/state`

The message layout and topic names come from Unitree's official sources:
- unitree_sdk2_python
- dfx_inspire_service
"""

from dataclasses import dataclass, field

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from cyclonedds.core import DDSException
from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.qos import Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

CHANNEL_CONFIG_HAS_INTERFACE = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
      </Interfaces>
    </General>
    <Tracing>
      <Verbosity>config</Verbosity>
      <OutputFile>/tmp/cdds.LOG</OutputFile>
    </Tracing>
  </Domain>
</CycloneDDS>"""

CHANNEL_CONFIG_AUTODETERMINE = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true" priority="default" multicast="default"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmd_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorCmd_"):
    mode: types.uint8 = 0
    q: types.float32 = 0.0
    dq: types.float32 = 0.0
    tau: types.float32 = 0.0
    kp: types.float32 = 0.0
    kd: types.float32 = 0.0
    reserve: types.array[types.uint32, 3] = field(default_factory=lambda: [0, 0, 0])


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorCmds_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorCmds_"):
    cmds: types.sequence[MotorCmd_] = field(default_factory=lambda: [])


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorState_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorState_"):
    mode: types.uint8 = 0
    q: types.float32 = 0.0
    dq: types.float32 = 0.0
    ddq: types.float32 = 0.0
    tau_est: types.float32 = 0.0
    q_raw: types.float32 = 0.0
    dq_raw: types.float32 = 0.0
    ddq_raw: types.float32 = 0.0
    temperature: types.uint8 = 0
    lost: types.uint32 = 0
    reserve: types.array[types.uint32, 2] = field(default_factory=lambda: [0, 0])


@dataclass
@annotate.final
@annotate.autoid("sequential")
class MotorStates_(idl.IdlStruct, typename="unitree_go.msg.dds_.MotorStates_"):
    states: types.sequence[MotorState_] = field(default_factory=lambda: [])


class UnitreeDdsSession:
    """Minimal DDS session for the Inspire hand topics."""

    def __init__(
        self,
        domain_id: int = 0,
        network_interface: str | None = None,
        command_topic: str = "rt/inspire/cmd",
        state_topic: str = "rt/inspire/state",
        qos: Qos | None = None,
    ) -> None:
        config = CHANNEL_CONFIG_AUTODETERMINE
        if network_interface:
            config = CHANNEL_CONFIG_HAS_INTERFACE.replace("$__IF_NAME__$", network_interface)

        self._domain = Domain(domain_id, config)
        self._participant = DomainParticipant(domain_id)
        self._command_topic = Topic(self._participant, command_topic, MotorCmds_, qos)
        self._state_topic = Topic(self._participant, state_topic, MotorStates_, qos)
        self._writer = DataWriter(self._participant, self._command_topic, qos)
        self._reader = DataReader(self._participant, self._state_topic, qos)

    def write(self, sample: MotorCmds_) -> None:
        self._writer.write(sample)

    def read_state(self, timeout_seconds: float | None = None) -> MotorStates_ | None:
        try:
            if timeout_seconds is None:
                return self._reader.take_one()
            return self._reader.take_one(timeout=duration(seconds=timeout_seconds))
        except TimeoutError:
            return None
        except DDSException as exc:
            raise RuntimeError(f"Failed to read Inspire hand state over DDS: {exc.msg}") from exc
