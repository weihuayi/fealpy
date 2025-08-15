import time
from enum import IntEnum
from typing import Iterable, Any
from collections import deque
from collections.abc import Callable
import threading

from ._types import NodeExceptionData
from .node import CNode, VariableOutput

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


class Graph():
    KEY_OF_NODE = "cnode"
    KEY_OF_EDGE = "conn"

    status : GraphStatus
    context : dict[Any, Any]
    output_node : VariableOutput
    error_listeners : list[Callable[[NodeExceptionData], Any]]
    status_listeners : list[Callable[[GraphStatus], Any]]

    def __init__(self):
        self.status = GraphStatus.READY
        self.context = {}
        self.output_node = VariableOutput({})
        self.controller = threading.Condition()
        self.error_listeners = []
        self.status_listeners = []

    @staticmethod
    def receive_data(context, node: CNode, input_values: dict[str, Any] = {}):
        for input_name, input_slot in node.input_slots.items():
            param_name = input_slot.param_name
            if input_slot.is_connected():
                for src_addr in input_slot.source_list:
                    context_key = src_addr.node.dump_key(src_addr.slot)
                    assert context_key in context
                    data = context[context_key]
                    if input_slot.variable:
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
                    raise RuntimeError(f"The '{input_name}' input of node "
                                    f"'{node}' is not connected "
                                    "and not having a default value.")

        return input_values

    @staticmethod
    def send_data(context, node: CNode, results: tuple[Any, ...]):
        if not isinstance(results, tuple):
            results = (results,)

        if len(results) >= 2 and len(results) != len(node.output_slots):
            raise RuntimeError("The number of results does not match "
                               "the number of output slots.")

        for output_slot, result in zip(node.output_slots.keys(), results):
            context_key = node.dump_key(output_slot)
            context[context_key] = result

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
                positional_inputs=args,
                keyword_inputs=kwargs
            )

    def execute(self) -> None:
        if self.status != GraphStatus.READY:
            raise GraphStatusError("Can not execute a graph with status %s".format(self.status))

        for node in self._topological_sort([self.output_node]):
            with self.controller:
                if self.status in (GraphStatus.STOP, GraphStatus.DEBUG):
                    break
                elif self.status == GraphStatus.PAUSE:
                    self.controller.wait()

            input_values = {}
            try:
                Graph.receive_data(self.context, node, input_values)
                results = node.run(**input_values)
                Graph.send_data(self.context, node, results)
            except Exception as error:
                with self.controller:
                    self._set_status(GraphStatus.DEBUG)
                error_info = self._capture_error(node, error, (), input_values)
                self._send_exception(error_info)
                break
        else:
            with self.controller:
                self._set_status(GraphStatus.STOP)

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
                self._set_status(GraphStatus.READY)

    def get(self) -> dict[str, Any]:
        return self.output_node.out

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


WORLD_GRAPH = Graph()
