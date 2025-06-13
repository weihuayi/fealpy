from typing import Set, Dict, List, Iterable, Any
from collections import deque

from .base import OutputSlot, Node, OutputNode

__all__ = ["Graph", "WORLD_GRAPH"]


class Graph():
    context : Dict[OutputSlot, Any]
    output_nodes : Dict[str, OutputNode]

    def __init__(self):
        self.output_nodes = {}
        self.context = {}

    def execute(self):
        for node in self._topological_sort(self.output_nodes.values()):
            node.execute(self.context)

        return {key: node.value for key, node in self.output_nodes.items()}

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

            for in_slot in current._input_slots.values():
                if in_slot.source is not None:
                    source_node = in_slot.source.node
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

    def OutputPort(self, name: str):
        if name in self.output_nodes:
            return self.output_nodes[name].IN
        else:
            node = OutputNode()
            self.output_nodes[name] = node
            return node.IN

    __getitem__ = OutputPort


WORLD_GRAPH = Graph()