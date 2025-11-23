
import importlib
from types import ModuleType
from typing import Any
from collections.abc import Iterator

from .core import Graph, CNode, NodeGroup
from .nodetype import CNodeType

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
        TITLE = getattr(nodetype, "TITLE", None)
        if TITLE and TITLE.lower().startswith(filter.lower()):
            yield {
                "name": nodetype.__name__,
                "title": TITLE,
                "path": getattr(nodetype, "PATH", ""),
                "desc": getattr(nodetype, "DESC", ""),
            }


def search_node(name: str):
    """Find a node by name."""
    from dataclasses import asdict

    register_all_nodes()

    if name in CNodeType.REGISTRY:
        nodetype = CNodeType.REGISTRY[name]
        inputs, outputs = [], []
        var_in, var_out = False, False

        for slot in nodetype.INPUT_SLOTS:
            if slot.name.startswith("*"):
                var_in = True
                continue
            inputs.append(asdict(slot))

        for slot in nodetype.OUTPUT_SLOTS:
            if slot.name.startswith("*"):
                var_out = True
                continue
            outputs.append(asdict(slot))

        return {
            "title": nodetype.TITLE,
            "inputs": inputs,
            "outputs": outputs,
            "var_in": var_in,
            "var_out": var_out,
        }

    raise ValueError(f"Node {name} not found")


def is_group_input(cnode_item: dict[str, Any]):
    return cnode_item.get("gin", None) is not None

def is_group_output(cnode_item: dict[str, Any]):
    return cnode_item.get("gout", None) is not None

def is_group(cnode_item: dict[str, Any]):
    return cnode_item.get("ref", None) is not None

def is_output_slot(slot_item: dict[str, Any]):
    return bool(slot_item["src"])


def load(data: dict, /, return_graph=True) -> CNode | NodeGroup | Graph:
    from .nodetype import create
    from .core.edge import connect_from_address, AddrHandler

    cnode_table, slots_table, conns_table = data["cnodes"], data["slots"], data["conns"]
    del data
    # Same order as the data
    cnode_list: list[CNode | NodeGroup | None] = []
    graph_dict: dict[int, Graph] = {}

    # STEP 1: Create cnode_list and graph_dict
    for cnode_item in cnode_table:
        # Skip graph IO nodes
        if is_group_input(cnode_item) or is_group_output(cnode_item):
            # use a placeholder to keep index, as the graph may not be created yet
            cnode_list.append(None)
            continue
        if is_group(cnode_item):
            graph_id = cnode_item["ref"]
            if graph_id not in graph_dict:
                graph_dict[graph_id] = Graph(cnode_item["name"])
            cnode_list.append(None)
            continue

        cnode_list.append(create(cnode_item["name"]))

    # STEP 2: Find graph IO slots to register source and drain for graphs.
    for slot_item in slots_table:
        cnode_id = slot_item["cnode"]
        cnode_item = cnode_table[cnode_id]

        if is_group_input(cnode_item):
            graph = graph_dict[cnode_item["gin"]]
            graph.register_source(slot_item["name"])

        elif is_group_output(cnode_item): # input slot of the output node of a graph
            graph = graph_dict[cnode_item["gout"]]
            graph.register_drain(slot_item["name"], default=slot_item["val"])

    # STEP 3: Generate actual graph IO nodes from graphs, and groups from graphs.
    for idx, cnode_item in enumerate(cnode_table):
        if is_group_input(cnode_item):
            graph = graph_dict[cnode_item["gin"]]
            assert cnode_list[idx] is None
            cnode_list[idx] = graph._source_node
        elif is_group_output(cnode_item):
            graph = graph_dict[cnode_item["gout"]]
            assert cnode_list[idx] is None
            cnode_list[idx] = graph._drain_node
        elif is_group(cnode_item):
            graph = graph_dict[cnode_item["ref"]]
            assert cnode_list[idx] is None
            cnode_list[idx] = NodeGroup(graph)

    # STEP 4: Set all inputs and their defaults
    for slot_item in slots_table:
        if is_output_slot(slot_item):
            continue
        cnode_id = slot_item["cnode"]
        cnode = cnode_list[cnode_id]
        try:
            cnode(**{slot_item["name"]: slot_item["val"]})
        except TypeError:
            pass

    # STEP 5: Recover connections
    for src_id, dst_id in ((item["src"], item["dst"]) for item in conns_table):
        src_item = slots_table[src_id]
        dst_item = slots_table[dst_id]
        src_node_id, src_name = src_item["cnode"], src_item["name"]
        dst_node_id, dst_name = dst_item["cnode"], dst_item["name"]
        src_node = cnode_list[src_node_id]
        dst_node = cnode_list[dst_node_id]
        # print(src_node, src_name, "->", dst_node, dst_name)
        try:
            addr = getattr(src_node(), src_name)
        except TypeError:
            addr = AddrHandler(src_node, src_name)

        try:
            dst_node(**{dst_name: addr})
        except TypeError:
            connect_from_address(dst_node.input_slots, {dst_name: addr})

    if return_graph and isinstance(cnode_list[0], NodeGroup):
        return cnode_list[0].graph

    return cnode_list[0]


def relation_table(group: NodeGroup, /):
    from queue import Queue
    from .core._types import CNode

    conn_list: list[tuple[tuple[CNode, str], tuple[CNode, str]]] = [] # [(src, dst)]
    output_map: dict[tuple[str, CNode], Any] = {} # {(name, node): None}
    input_map: dict[tuple[str, CNode], Any] = {} # {(name, node): default}
    cnode_map: dict[CNode, None] = {}
    graph_map: dict[Graph, None] = {}

    node_queue = Queue()
    node_queue.put(group)

    while not node_queue.empty():
        current_node = node_queue.get()

        if current_node in cnode_map:
            continue

        # branch into the graph
        if isinstance(current_node, NodeGroup):
            graph_map[current_node.graph] = None
            # The graph input node is also need to be included manually
            # node_queue.put(current_node._input_node)
            node_queue.put(current_node.graph._drain_node)

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


def dump(global_group: NodeGroup | Graph, /) -> dict[str, list[dict[str, Any]]]:
    if isinstance(global_group, Graph):
        global_group = NodeGroup(global_group)

    cl, om, im, nm, gm = relation_table(global_group)
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
        if isinstance(node, NodeGroup):
            gi, go = None, None
            if node.graph.name is None:
                name = "Graph_" + str(graph_idx[node.graph])
            else:
                name = node.graph.name

        elif isinstance(node, GraphInputNode):
            gi, go = graph_idx[node.graph], None
            if node.graph.name is None:
                name = "GroupInput({})".format("Graph " + graph_idx[node.graph])
            else:
                name = "GroupInput({})".format(node.graph.name)

        elif isinstance(node, GraphOutputNode):
            gi, go = None, graph_idx[node.graph]
            if node.graph.name is None:
                name = "GroupOutput({})".format("Graph " + graph_idx[node.graph])
            else:
                name = "GroupOutput({})".format(node.graph.name)

        else:
            assert hasattr(node, "__node_type__")
            gi, go = None, None
            name = node.__node_type__

        ref = graph_idx[node.graph] if isinstance(node, NodeGroup) else None
        cnode_table.append(
            {"name": name, "ref": ref, "gin": gi, "gout": go}
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
