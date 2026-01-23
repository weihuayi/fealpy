import time
from enum import IntEnum
from typing import Any
from collections import deque, OrderedDict
from collections.abc import Callable, Mapping, Iterable
import threading
import traceback

from . import edge as _E
from ._types import (
    CNode,
    NodeExceptionData,
    Slot, OutputSlot, TransSlot, InputSlot,
    NodeIOError
)

__all__ = ["Graph", "WORLD_GRAPH"]


class GraphStatusError(Exception):
    """Inappropriate status of graph."""
    pass


class GraphStatus(IntEnum):
    READY = 0
    RUN = 1
    PAUSE = 2
    DEBUG = 3
    STOP = 4


class GraphController:
    """Graph controller."""
    def __init__(self):
        self.condition = threading.Condition()
        self.status = GraphStatus.READY


class GraphCtxOps:
    @staticmethod
    def receive_data(
        context: dict[int, Any],
        rec_slots: Mapping[str, TransSlot],
        input_values: dict[str, Any] = {},
        node_msg: str = ""
    ):
        for input_name, input_slot in rec_slots.items():
            param_name = getattr(input_slot, "param_name", input_name)
            if input_slot.is_connected():
                for src_node, src_slot in input_slot.source_list:
                    context_key = id(src_node.output_slots[src_slot])
                    if context_key in context:
                        data = context[context_key]
                    else:
                        raise NodeIOError("Data not received from upstream nodes "
                                          f"when executing node '{node_msg}'.")

                    if getattr(input_slot, "variable", False):
                        if param_name not in input_values:
                            input_values[param_name] = []
                        input_values[param_name].append(data)
                    else:
                        input_values[param_name] = data
                        break
            else:
                if input_slot.has_default:
                    input_values[param_name] = input_slot.default
                else:
                    raise NodeIOError(f"The input '{input_name}' of node "
                                      f"'{node_msg}' is not connected "
                                      "and not having a default value.")

        return input_values

    @staticmethod
    def send_data(
        context: dict[int, Any],
        send_slots: Mapping[str, Slot],
        results: Any,
        node_msg: str = ""
    ):
        if results is None:
            return

        if not isinstance(results, tuple):
            results = (results,)

        if len(results) >= 2 and len(results) != len(send_slots):
            raise NodeIOError("The number of results does not match "
                              f"the number of output slots in the node '{node_msg}'.")

        for output_slot, result in zip(send_slots.values(), results):
            context_key = id(output_slot)
            if context_key in context:
                raise NodeIOError(f"Data sent by node {node_msg} already exists. "
                                  "Reset the context before executing the graph.")
            context[context_key] = result


class Graph:
    KEY_OF_NODE = "cnode"
    KEY_OF_EDGE = "conn"

    status : GraphStatus
    _source_slots : OrderedDict[str, OutputSlot]
    _drain_slots : OrderedDict[str, InputSlot]
    context : dict[int, Any]
    error_listeners : list[Callable[[NodeExceptionData], Any]]
    status_listeners : list[Callable[[GraphStatus], Any]]

    def __init__(self, name: str | None = None, /):
        self.name = name
        self.status = GraphStatus.READY
        self.context = {}
        self.controller = threading.Condition()
        self.error_listeners = []
        self.status_listeners = []
        self._source_slots = OrderedDict()
        self._drain_slots = OrderedDict()
        self._source_node = GraphInputNode(self, self._source_slots)
        self._drain_node = GraphOutputNode(self, self._drain_slots)
        self._log_message = []

    def __repr__(self):
        if self.name is None:
            return "$$anonymous graph at " + hex(id(self)) + "$$"
        return "$$" + self.name + "$$"

    def __bool__(self) -> bool:
        return True

    def register_source(self, name: str, /):
        self._source_slots[name] = OutputSlot()

    def register_drain(self, name: str, /, **kwargs):
        self._drain_slots[name] = InputSlot(
            has_default="default" in kwargs,
            default=kwargs.get("default", None),
            param_name=name
        )

    def input(self, slot: str, /):
        if slot not in self._source_slots:
            self.register_source(slot)
        return _E.AddrHandler(self._source_node, slot)

    def output(self, **kwargs: _E.AddrHandler):
        for name in kwargs.keys():
            if name not in self._drain_slots:
                self.register_drain(name)
        _E.connect_from_address(self._drain_slots, kwargs)

    # def requests(self):
    #     request_set: set[CNode] = set()

    #     for outslot in self.output_slots.values():
    #         request_set = request_set.union(
    #             set(addr.node for addr in outslot.source_list)
    #         )

    #     return request_set

    def _send_exception(self, info: NodeExceptionData) -> None:
        for callback in self.error_listeners:
            callback(info)

    def _set_status(self, status: GraphStatus) -> None:
        if status != self.status:
            self.status = status
            for callback in self.status_listeners:
                callback(self.status)

    @staticmethod
    def _capture_error(node: CNode, error: Exception, args, kwargs) -> NodeExceptionData:
        return NodeExceptionData(
                node=node,
                timestamp=time.time(),
                errtype=type(error),
                message=str(error),
                traceback=traceback.format_exc(),
                positional_inputs=args,
                keyword_inputs=kwargs
            )

    def execute(self, source: dict[str, Any] = {}, capture_err=True) -> None:
        if self.status != GraphStatus.READY:
            raise GraphStatusError("Can not execute a graph with status %s".format(self.status))

        for key in self._source_slots:
            if key not in source:
                raise NodeIOError(f"The input {key!r} is not provided.")
            self._source_node.data[key] = source[key]

        for node in self._topological_sort([self._drain_node]):
            with self.controller:
                if self.status in (GraphStatus.STOP, GraphStatus.DEBUG):
                    break
                elif self.status == GraphStatus.PAUSE:
                    self.controller.wait()

            input_values = {}
            try:
                GraphCtxOps.receive_data(self.context, node.input_slots, input_values, repr(node))
                results = node.run(**input_values)
                GraphCtxOps.send_data(self.context, node.output_slots, results, repr(node))
            except Exception as error:
                if not capture_err:
                    raise NodeIOError(f"node {node!r} execution failed") from error
                with self.controller:
                    self._set_status(GraphStatus.DEBUG)
                error_info = self._capture_error(node, error, (), input_values)
                self._send_exception(error_info)
                break

    def pause(self):
        """Pause or resume the graph execution."""
        with self.controller:
            if self.status == GraphStatus.RUN:
                self._set_status(GraphStatus.PAUSE)

            elif self.status == GraphStatus.PAUSE:
                self._set_status(GraphStatus.RUN)
                self.controller.notify()

            else:
                raise GraphStatusError("Graph is not running or paused")

    def stop(self):
        with self.controller:
            self._set_status(GraphStatus.STOP)

    def reset(self) -> None:
        with self.controller:
            if self.status == GraphStatus.STOP:
                self.context.clear()
                self._source_node.data.clear()
                self._drain_node.data.clear()
                self._set_status(GraphStatus.READY)

    def get(self) -> dict[str, Any]:
        return self._drain_node.data.copy()

    @staticmethod
    def _collect_relevant_nodes(nodes: Iterable[CNode]):
        stack = list(nodes)
        visited   : set[CNode]              = set()
        in_degree : dict[CNode, int]        = {}
        adj_list  : dict[CNode, list[CNode]] = {}

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            in_degree[current] = 0

            if current not in adj_list:
                adj_list[current] = []

            for in_slot in current.input_slots.values():
                for source_addr in in_slot.source_list:
                    source_node = source_addr.node
                    stack.append(source_node)
                    in_degree[current] += 1

                    if source_node not in adj_list:
                        adj_list[source_node] = []

                    adj_list[source_node].append(current)

        return in_degree, adj_list

    @staticmethod
    def _topological_sort(nodes: Iterable[CNode]) -> list[CNode]:
        in_degree, adj_list = Graph._collect_relevant_nodes(nodes)

        queue = deque(node for node, degree in in_degree.items() if degree == 0)
        sorted_nodes = []

        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_nodes) != len(adj_list):
            raise ValueError("There is a circular dependency in the computation graph.")

        return sorted_nodes

    def register_error_hook(self, callback: Callable[[NodeExceptionData], Any]):
        self.error_listeners.append(callback)

    def register_status_change_hook(self, callback: Callable[[GraphStatus], Any]):
        self.status_listeners.append(callback)

    def new_log_handler(self):
        from .loghandler import StringListHandler
        return StringListHandler(self._log_message)

    def log_message(self) -> list[str]:
        return self._log_message

    def clear_log_message(self):
        self._log_message.clear()


class GraphInputNode:
    def __init__(self, graph: Graph, outputs, /):
        self.graph = graph
        self.data = OrderedDict()
        self._slots = outputs

    def __repr__(self):
        if self.graph.name is None:
            return "$graph input node at " + hex(id(self)) + "$"
        else:
            return "$input node (" + self.graph.name + ")$"

    @property
    def input_slots(self): return {}
    @property
    def output_slots(self): return self._slots

    def run(self):
        result = tuple(self.data.values())
        self.data.clear()
        return result


class GraphOutputNode:
    def __init__(self, graph: Graph, inputs, /):
        self.graph = graph
        self.data = OrderedDict()
        self._slots = inputs

    def __repr__(self):
        if self.graph.name is None:
            return "$graph output node at " + hex(id(self)) + "$"
        else:
            return "$output node (" + self.graph.name + ")$"

    @property
    def input_slots(self): return self._slots
    @property
    def output_slots(self): return {}

    def run(self, **kwargs):
        self.data.clear()
        self.data.update(kwargs)


WORLD_GRAPH = Graph("WORLD_GRAPH")
