from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Protocol, NamedTuple, Any
from collections.abc import Mapping

__all__ = ["SlotStatus", "InputSlot", "OutputSlot",
           "CNode", "NodeExceptionData", "NodeIOError", "NodeTopologyError"]


class ParamPassingKind(IntEnum):
    POSITIONAL     = 0
    KEYWORD        = 1
    VAR_POSITIONAL = 2
    VAR_KEYWORD    = 3


class SlotStatus(IntEnum):
    ACTIVE = auto()
    BLOCKED = auto()
    DISABLED = auto()


class SourceAddr(NamedTuple):
    """A named tuple containing the source node and slot name."""
    node: "CNode"
    slot: str


@dataclass(slots=True)
class Slot():
    status: SlotStatus = SlotStatus.ACTIVE

    def is_active(self) -> bool:
        return self.status == SlotStatus.ACTIVE

    def is_blocked(self) -> bool:
        return self.status == SlotStatus.BLOCKED

    def is_disabled(self) -> bool:
        return self.status == SlotStatus.DISABLED


@dataclass(slots=True)
class TransSlot(Slot):
    has_default : bool             = False
    default     : Any              = None
    source_list : list[SourceAddr] = field(default_factory=list, init=False, compare=False)

    def is_connected(self) -> bool:
        return len(self.source_list) > 0

    def connect(self, source: "CNode", slot: str):
        if not self.is_connected():
            self.source_list.append(SourceAddr(source, slot))
        else:
            raise NodeTopologyError("multiple connections to non-variable input")


@dataclass(slots=True)
class InputSlot(TransSlot):
    variable   : bool       = False
    param_name : str | None = None

    def is_connected(self) -> bool:
        return len(self.source_list) > 0

    def connect(self, source: "CNode", slot: str):
        if self.variable or not self.is_connected():
            self.source_list.append(SourceAddr(source, slot))
        else:
            raise NodeTopologyError("multiple connections to non-variable input")


@dataclass(slots=True)
class OutputSlot(Slot):
    pass


class CNode(Protocol):
    @property
    def input_slots(self) -> Mapping[str, InputSlot]: ...
    @property
    def output_slots(self) -> Mapping[str, OutputSlot]: ...
    def run(self, *args, **kwargs) -> Any: ...


@dataclass(frozen=True, slots=True)
class NodeExceptionData():
    node : CNode
    timestamp : float = 0.
    errtype : type[Exception] = Exception
    message : str | None = None
    positional_inputs : list[Any] = field(default_factory=list)
    keyword_inputs : dict[str, Any] = field(default_factory=dict)


class NodeIOError(Exception):
    """Inappropriate input or output of nodes."""
    pass


class NodeTopologyError(Exception):
    """Inappropriate connection."""
    pass
