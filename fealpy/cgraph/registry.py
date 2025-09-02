
import importlib
from types import ModuleType
from typing import Any
from collections.abc import Iterator

from .core import Graph, CNode
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


def load(data: dict[str, list[dict[str, Any]]], /) -> Graph | CNode:
    from .nodetype import create
    from .core.edge import connect_from_address, AddrHandler

    num_graphs = max(cnode["ref"] for cnode in data["cnodes"]) + 1
    cnode_table, slots_table, conns_table = data["cnodes"], data["slots"], data["conns"]
    del data

    # Same order as the data
    cnode_list = []
    graph_list = [None] * num_graphs

    # Create cnodes and graphs
    for cnode_item in cnode_table:
        # Skip graph IO nodes
        if cnode_item["gi"] > -1 or cnode_item["go"] > -1:
            # use a placeholder to keep index, as the graph may not be created yet
            cnode_list.append(None)
            continue

        if cnode_item["ref"] == -1:
            cnode = create(cnode_item["name"])
        else:
            cnode = Graph(cnode_item["name"])
            graph_list[cnode_item["ref"]] = cnode

        cnode_list.append(cnode)

    # Loop again to replace placeholders with actual graph IO nodes
    for idx, cnode_item in enumerate(cnode_table):
        if cnode_item["gi"] > -1:
            graph = graph_list[cnode_item["gi"]]
            assert cnode_list[idx] is None
            cnode_list[idx] = graph._input_node
        elif cnode_item["go"] > -1:
            graph = graph_list[cnode_item["go"]]
            assert cnode_list[idx] is None
            cnode_list[idx] = graph._output_node

    # Set inputs and their defaults
    for slot_item in slots_table:
        if not slot_item["src"]:
            cnode_id = slot_item["cnode"]
            cnode_item = cnode_table[cnode_id]

            # Register IO slots got subgraphs:
            # Operations are done through the graph, but not their IO nodes which
            # are only wrappers.
            if cnode_item["go"] > -1: # input slot of the output node of a graph
                graph = graph_list[cnode_item["go"]]
                graph.register_output(slot_item["name"], default=slot_item["val"])
                graph.output_slots[slot_item["name"]].default = slot_item["val"]
            else:
                cnode = cnode_list[cnode_id]
                if cnode_item["ref"] > -1: # input slot of a subgraph
                    cnode.register_input(slot_item["name"], default=slot_item["val"])
                cnode.input_slots[slot_item["name"]].default = slot_item["val"]

    # Recover connections
    for src_id, dst_id in ((item["src"], item["dst"]) for item in conns_table):
        src_item = slots_table[src_id]
        dst_item = slots_table[dst_id]
        src_node_id, src_name = src_item["cnode"], src_item["name"]
        dst_node_id, dst_name = dst_item["cnode"], dst_item["name"]
        src_node = cnode_list[src_node_id]
        dst_node = cnode_list[dst_node_id]
        # print(src_node, src_name, "->", dst_node, dst_name)
        connect_from_address(dst_node.input_slots, {dst_name: AddrHandler(src_node, src_name)})

    return cnode_list[0]


def relation_table(graph: Graph, /):
    from queue import Queue
    from .core._types import CNode

    conn_list: list[tuple[tuple[CNode, str], tuple[CNode, str]]] = [] # [(src, dst)]
    output_map: dict[tuple[str, CNode], Any] = {} # {(name, node): None}
    input_map: dict[tuple[str, CNode], Any] = {} # {(name, node): default}
    cnode_map: dict[CNode, None] = {}
    graph_map: dict[Graph, None] = {}

    node_queue = Queue()
    node_queue.put(graph)

    while not node_queue.empty():
        current_node = node_queue.get()

        if current_node in cnode_map:
            continue

        # branch into the graph
        if isinstance(current_node, Graph):
            graph_map[current_node] = None
            # The graph input node is also need to be included manually
            node_queue.put(current_node._input_node)
            node_queue.put(current_node._output_node)

        cnode_map[current_node] = None
        # collect all inputs
        for name, in_slot in current_node.input_slots.items():
            input_map[name, current_node] = in_slot.default
            # search upstreams and connections
            for src_node, src_slot in in_slot.source_list:
                # put upstream into the stack
                node_queue.put(src_node)
                # record connection
                conn_list.append(((src_node, src_slot), (current_node, name)))

        # collect all outputs
        for name, out_slot in current_node.output_slots.items():
            # Defaults of out_slots are not needed, because:
            # (1) for typed cnodes, outputs has no defaults (OutputSlot)
            # (2) for a node group, defaults of outputs can be managed by its output node
            # (3) for a graph input node, defaults of outputs can be managed by the graph
            output_map[name, current_node] = None

    return conn_list, output_map, input_map, cnode_map, graph_map


def dump(global_graph: Graph, /):
    cl, om, im, nm, gm = relation_table(global_graph)
    # make index
    output_idx = {key: idx for idx, key in enumerate(om.keys())}
    input_idx = {key: idx for idx, key in enumerate(im.keys())}
    cnode_idx = {key: idx for idx, key in enumerate(nm.keys())}
    graph_idx = {key: idx for idx, key in enumerate(gm.keys())}

    conn_table = []
    slot_table = []
    cnode_table = []

    for node, _ in nm.items():
        from .core.graph import GraphInputNode, GraphOutputNode
        if isinstance(node, Graph):
            gi, go = -1, -1
            if node.name is None:
                name = "Graph " + graph_idx[node]
            else:
                name = node.name

        elif isinstance(node, GraphInputNode):
            gi, go = graph_idx[node.graph], -1
            if node.graph.name is None:
                name = "GroupInput({})".format("Graph " + graph_idx[node.graph])
            else:
                name = "GroupInput({})".format(node.graph.name)

        elif isinstance(node, GraphOutputNode):
            gi, go = -1, graph_idx[node.graph]
            if node.graph.name is None:
                name = "GroupOutput({})".format("Graph " + graph_idx[node.graph])
            else:
                name = "GroupOutput({})".format(node.graph.name)

        else:
            assert hasattr(node, "__node_type__")
            gi, go = -1, -1
            name = node.__node_type__

        ref = graph_idx[node] if isinstance(node, Graph) else -1
        cnode_table.append(
            {"name": name, "ref": ref, "gi": gi, "go": go}
        )

    for (name, cnode), default in im.items():
        slot_table.append(
            {"src": False, "cnode": cnode_idx[cnode], "name": name, "val": default}
        )

    for (name, cnode), default in om.items():
        slot_table.append(
            {"src": True, "cnode": cnode_idx[cnode], "name": name, "val": default}
        )

    for (src_node, src_slot), (dst_node, dst_slot) in cl:
        conn_table.append(
            {
                "src": output_idx[src_slot, src_node] + len(im), # bias
                "dst": input_idx[dst_slot, dst_node],
            }
        )

    return {
        "conns": conn_table,
        "slots": slot_table,
        "cnodes": cnode_table
    }
