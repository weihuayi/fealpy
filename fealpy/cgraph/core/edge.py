from collections.abc import Mapping

from ._types import CNode, InputSlot, NodeTopologyError

__all__ = ["connect", "AddrHandler", "connect_from_address"]


class AddrHandler:
    __slots__ = ("_node", "_slot")

    def __init__(self, node: CNode, slot: str | None):
        self._node = node
        self._slot = slot

    def __getattr__(self, name: str):
        self._slot = name
        return self

    def _take_the_first_port(self) -> None:
        for name in self._node.output_slots.keys():
            self._slot = name
            break
        else:
            raise NodeTopologyError(f"No output slot for node {self._node}")

    def get_addr(self) -> tuple[CNode, str]:
        if self._slot is None:
            self._take_the_first_port()
        elif self._slot not in self._node.output_slots:
            raise NodeTopologyError(f"Output slot {self._slot} does not exist")

        return self._node, self._slot


def connect_from_address(
        input_slots: Mapping[str, InputSlot],
        addr_kwds: Mapping[str, AddrHandler | tuple[AddrHandler]]
) -> None:
    for name, addrs in addr_kwds.items():
        if isinstance(addrs, (tuple, list)):
            is_addr_handler = [isinstance(addr, AddrHandler) for addr in addrs]
            if not any(is_addr_handler): # a default value in type tuple or list
                addrs = (addrs, )
            elif all(is_addr_handler): # a tuple or list of AddrHandler
                pass
            else:
                raise NodeTopologyError("Mixed types of output address and default values")
        else:
            addrs = (addrs, )

        if name in input_slots:
            in_slot = input_slots[name]
            for addr in addrs:
                if not isinstance(addr, AddrHandler):
                    in_slot.default = addr
                else:
                    source_node, source_slot = addr.get_addr()
                    in_slot.connect(source_node, source_slot)

        else:
            raise NodeTopologyError(f"Input slot {name} does not exist")
