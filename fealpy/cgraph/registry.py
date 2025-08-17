
import importlib
from types import ModuleType
from typing import Any
from collections.abc import Iterator

from .core import Graph
from .nodetype import CNodeType, to_dict, from_dict

__all__ = ["register_all_nodes", "search_all_nodes", "search_node", "load", "dump"]


SEARCH_ROOT = "fealpy.cgraph"


# find all __nodes__ recursively
def all_nodes(path: str) -> Iterator[Any]:
    m = importlib.import_module(path)

    if isinstance(m, ModuleType) and "__nodes__" in m.__dict__:
        for n in m.__nodes__:
            yield from all_nodes(".".join([path, n]))
    else:
        yield m


def register_all_nodes(path=SEARCH_ROOT):
    for _ in all_nodes(path):
        pass


def search_all_nodes(filter: str = ""):
    """Search all nodes."""
    register_all_nodes()

    for nodetype in CNodeType.REGISTRY.values():
        TITLE = nodetype.TITLE
        if TITLE and TITLE.lower().startswith(filter.lower()):
            yield {
                "name": nodetype.__name__,
                "title": TITLE,
                "path": nodetype.PATH,
            }


def search_node(name: str):
    """Find a node by name."""
    from dataclasses import asdict

    register_all_nodes()

    if name in CNodeType.REGISTRY:
        nodetype = CNodeType.REGISTRY[name]

        return {
            "title": nodetype.TITLE,
            "inputs": [asdict(d) for d in nodetype.INPUT_SLOTS],
            "outputs": [asdict(d) for d in nodetype.OUTPUT_SLOTS],
            "variable": nodetype.VARIABLE,
        }

    raise ValueError(f"Node {name} not found")


def _check_src_dst(data) -> tuple[int, str]:
    if not isinstance(data, tuple) or len(data) != 2:
        raise TypeError("src and dst must be both 2-tuples")

    if not isinstance(data[0], int): # node index
        raise TypeError("node index must be an integer")
    if not isinstance(data[1], str): # slot name
        raise TypeError("slot name must be a string")

    return data


def load(data: dict):
    nodes = []
    NODE_KEY = Graph.KEY_OF_NODE
    EDGE_KEY = Graph.KEY_OF_EDGE
    graph = Graph()

    assert NODE_KEY in data, f"node data as key '{NODE_KEY}' not found"
    assert isinstance(data[NODE_KEY], list), "node data must be a list"
    assert EDGE_KEY in data, f"edge data as key '{EDGE_KEY}' not found"
    assert isinstance(data[EDGE_KEY], list), "edge data must be a list"
    nodes.append(graph.output_node)

    for node_data in data[NODE_KEY]:
        nodes.append(from_dict(node_data))

    for edge_data in data[EDGE_KEY]:
        assert isinstance(edge_data, dict), "data of each edge must be a dict"
        assert "src" in edge_data, "edge data must have key 'src'"
        assert "dst" in edge_data, "edge data must have key 'dst'"
        src_node_id, outslot = _check_src_dst(tuple(edge_data["src"]))
        dst_node_id, inslot = _check_src_dst(tuple(edge_data["dst"]))

        if (src_node_id >= len(nodes)) or (src_node_id < 1):
            # NOTE: src_node_id = 0 is the output node of graph, which should
            # not be a source node.
            raise ValueError(f"src node ID {src_node_id} is out of range")
        if (dst_node_id >= len(nodes)) or (dst_node_id < 0):
            raise ValueError(f"dst node ID {dst_node_id} is out of range")

        src_node = nodes[src_node_id]
        dst_node = nodes[dst_node_id]
        dst_node(**{inslot: getattr(src_node(), outslot)})

    return graph


def dump(graph: Graph):
    stack = [graph.output_node]
    indices = {graph.output_node: 0}
    nodes = []
    edges = []

    while stack:
        current = stack.pop()

        for name, in_slot in current.input_slots.items():
            for source_addr in in_slot.source_list:
                source_node = source_addr.node

                if source_node not in indices:
                    indices[source_node] = len(indices)
                    nodes.append(to_dict(source_node))
                    stack.append(source_node)

                edges.append(
                    {
                        "src": [indices[source_node], source_addr.slot],
                        "dst": [indices[current], name]
                    }
                )

    return {
        graph.KEY_OF_NODE: nodes,
        graph.KEY_OF_EDGE: edges
    }
