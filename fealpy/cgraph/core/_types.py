from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Protocol, runtime_checkable, NamedTuple, Any
from collections.abc import Mapping


class DataType(IntEnum):
    NONE = auto()
    STRING = auto()
    INT = auto()
    FLOAT = auto()
    TENSOR = auto()
    BOOL = auto()
    GEOMETRY = auto()
    SPACE = auto()


class SlotStatus(IntEnum):
    ACTIVE = auto()
    BLOCKED = auto()
    DISABLED = auto()


class SourceAddr(NamedTuple):
    """A named tuple containing the source node and slot name."""
    node: "Node"
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
class InputSlot(Slot):
    has_default: bool = False
    default: Any = None
    param_name: str | None = None
    variable: bool = False
    source_list: list[SourceAddr] = field(default_factory=list, init=False, compare=False)

    def is_connected(self) -> bool:
        return len(self.source_list) > 0

    def connect(self, source: "Node", slot: str):
        if self.variable or not self.is_connected():
            self.source_list.append(SourceAddr(source, slot))
        else:
            raise NodeTopologyError("multiple connections to non-variable input")


@dataclass(slots=True)
class OutputSlot(Slot):
    pass


@runtime_checkable
class Node(Protocol):
    @property
    def input_slots(self) -> Mapping[str, InputSlot]: ...
    @property
    def output_slots(self) -> Mapping[str, OutputSlot]: ...
    @property
    def is_variable(self) -> bool: ...
    def run(self, *args, **kwargs) -> Any: ...
    def dump_key(self, slot: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class NodeExceptionData():
    node : Node
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


class GraphStatusError(Exception):
    """Inappropriate status of graph."""
    pass


class GraphStatus(IntEnum):
    READY = auto()
    RUN = auto()
    DEBUG = auto()
    STOP = auto()
