from typing import Set, Dict, List, Iterable, Any
from collections import deque

from ._types import Node

__all__ = ["Graph", "WORLD_GRAPH"]


class Graph():
    context : dict[Any, Any]
    output_nodes : list[Node]

    def __init__(self):
        self.output_nodes = []
        self.context = {}

    @staticmethod
    def receive_data(context, node: Node, input_values: dict[str, Any] = {}):
        for input_name, input_slot in node.input_slots.items():
            if input_slot.is_connected():
                for src_addr in input_slot.source_list:
                    context_key = src_addr.node.dump_key(src_addr.slot)
                    assert context_key in context
                    data = context[context_key]
                    param_name = input_slot.param_name
                    if input_slot.variable:
                        if param_name not in input_values:
                            input_values[param_name] = []
                        input_values[param_name].append(data)
                    else:
                        input_values[param_name] = data
                        break
            else:
                if input_slot.has_default:
                    input_values[input_name] = input_slot.default
                else:
                    raise RuntimeError(f"The '{input_name}' input of node "
                                    f"'{node}' is not connected "
                                    "and not having a default value.")

        return input_values

    @staticmethod
    def send_data(context, node: Node, results: tuple[Any, ...]):
        if not isinstance(results, tuple):
            results = (results,)

        if len(results) != len(node.output_slots):
            raise RuntimeError()

        for output_slot, result in zip(node.output_slots.keys(), results):
            context_key = node.dump_key(output_slot)
            context[context_key] = result

    def execute(self) -> None:
        for node in self._topological_sort(self.output_nodes.values()):
            input_values = {}
            Graph.receive_data(self.context, node, input_values)
            results = node.execute(**input_values)
            Graph.send_data(self.context, node, results)

    def pause(self):
        pass

    def stop(self):
        pass

    def reset(self) -> None:
        self.context.clear()

    @staticmethod
    def _collect_relevant_nodes(nodes: Iterable[Node]):
        stack = list(nodes)
        visited   : Set[Node]              = set()
        in_degree : Dict[Node, int]        = {}
        adj_list  : Dict[Node, List[Node]] = {}

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
    def _topological_sort(nodes: Iterable[Node]) -> List[Node]:
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

    def register_output_node(self, node: Node):
        self.output_nodes.append(node)


WORLD_GRAPH = Graph()
