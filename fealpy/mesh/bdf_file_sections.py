import re
from typing import List, Tuple, Dict, Type, Optional, Any
from fealpy.backend import bm


def parse_fixed_format(line: str):
    """按固定格式解析（每字段 8 字符）"""
    fields = [line[i:i + 8].strip() for i in range(0, min(len(line), 80), 8)]
    fields = [f for f in fields if f]  # 移除空字段
    return fields

def parse_free_format(line: str):
    """按自由格式解析（逗号分隔）"""
    fields = [f.strip() for f in line.split(',')]
    fields = [f for f in fields if f]  # 移除空字段
    return fields


data_map = {
    'NODE': ['node_ids', ['nodes', 'xyz']],
    'ELEMENT': ['element_ids', ['elements', 'type'], ['elements', 'node_ids']]
}

mesh_type_map = {
    'CTRIA3': 'triangle',
    'CQUAD4': 'quadrangle',
    'CTETRA': 'tetrahedron',
    'CHEXA': 'hexahedron'
}

class Section:
    """
    Abstract base class for all parsed Nastran *.bdf file sections.

    Each subclass represents a different section keyword and is responsible for parsing
    corresponding lines and attaching data to the mesh representation.

    Parameters:
        options (Dict[str, str]): Parsed keyword arguments from the section header.

    Attributes:
        options (Dict[str, str]): Stores parsed options for the section.
        keyword (str): Section keyword (e.g., 'NODE', 'ELEMENT'). Must be defined by subclasses.

    Methods:
        match_keyword(keyword: str) -> bool:
            Checks whether the input keyword matches this section's keyword.

        parse_line(line: str) -> None:
            Parses a line within this section. Must be implemented by subclasses.

        attach(meshdata: Dict[str, Any]) -> None:
            Attaches parsed data to the meshdata dictionary (optional override).

        finalize() -> None:
            Finalizes section data (e.g., converts lists to arrays, builds indices).
    """
    keyword: str = ''

    def __init__(self, options: Dict[str, str]):
        self.options = options

    @classmethod
    def match_keyword(cls, keyword: str) -> bool:
        return cls.keyword.upper() == keyword.upper()

    def parse_line(self, line: str, is_free_format: bool=True) -> None:
        raise NotImplementedError(f"parse_line must be implemented by {self.__class__.__name__}")

    def attach(self, meshdata: Dict[str, Any]):
        pass

    def finalize(self) -> None:
        """
        Optional: convert stored lists to bm.array
        """
        pass


class NodeSection(Section):
    """
    Parses the NODE section from a Nastran .bdf file.

    This section defines the global node ID and coordinates for each mesh node.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header.

    Attributes:
        _id (List[int]): Node IDs.
        _node (List[List[float]]): Node coordinates (as nested lists).
        id (bm.array | None): Node IDs as array.
        node (bm.array | None): Node coordinates as array.
        node_map (bm.array | None): Mapping from node ID to index in the node array.

    Methods:
        parse_line(line: str): Parses a single line of node data.
        finalize(): Converts internal storage to bm.array and builds index mapping.
        attach(meshdata: Dict[str, Any]): Adds node map to shared meshdata dictionary.
    """
    keyword = 'NODE'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self._id: List[int] = []
        self._node: List[List[float]] = []
        self.id: Optional[Any] = None
        self.node: Optional[Any] = None
        self.node_map: Optional[Any] = None


    def parse_line(self, line: str, is_free_format: bool=True) -> None:
        if is_free_format:
            parts = parse_free_format(line)
        else:
            parts = parse_fixed_format(line)
        nid = int(parts[1])
        coords = [float(val) for val in parts[2:]]
        self._id.append(nid)
        self._node.append(coords)

    def finalize(self) -> None:
        self.id = bm.array(self._id)
        self.node = bm.array(self._node)
        N = bm.max(self.id) + 1
        self.node_map = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.node_map, self.id, bm.arange(len(self.id), dtype=bm.int32))

    def finalize_nastran(self, bdf) -> None:
        node_ = []
        node_id_ = []
        for node_id, node in bdf.nodes.items():
            node_id_.append(node_id)
            # node.xyz 包含 [x, y, z] 坐标
            node_.append([node.xyz[0], node.xyz[1], node.xyz[2]])
        self.id = bm.array(node_id_)
        self.node = bm.array(node_)
        N = bm.max(self.id) + 1
        self.node_map = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.node_map, self.id, bm.arange(len(self.id), dtype=bm.int32))

    def attach(self, meshdata: Dict[str, Any]) -> None:
        meshdata['node_map'] = self.node_map

class ElementSection(Section):
    """
    Parses the *ELEMENT section from a Nastran .bdf file.

    This section defines element connectivity using global node IDs. It stores both raw
    data and post-processed tensor structures for efficient access.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header.

    Attributes:
        _id (List[int]): Element IDs parsed from each line.
        _cell (Dict[str, List[List[int]]]): Element connectivity,
            where keyed by element type and each list contains the node IDs of an element.
        id (bm.array | None): Array of element IDs.
        cell (Dict[str, array] | None): Array of element connectivities, where keyed by element type.
        cell_map (bm.array | None): Index mapping from element ID to row index in the `cell` array.

    Methods:
        parse_line(line: str): Parses a single line of element connectivity data.
        finalize(): Converts internal storage to bm.array and builds element ID index mapping.
        attach(meshdata: Dict[str, Any]): Adds element map to shared meshdata dictionary.
    """
    keyword = 'ELEMENT'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self._id: List[int] = []
        self._cell: Dict[str, List[List[int]]] = {}
        # final arrays
        self.id: Optional[Any] = None
        self.cell: Optional[Any] = None
        self.cell_map: Optional[Any] = None

    def parse_line(self, line: str, is_free_format: bool=True) -> None:
        if is_free_format:
            parts = parse_free_format(line)
        else:
            parts = parse_fixed_format(line)
        etype = parts[0].upper()
        eid = int(parts[1])
        conn = [int(val) for val in parts[3:]]
        self._id.append(eid)
        self._cell.setdefault(etype, []).append(conn)

    def finalize(self) -> None:
        self.id = bm.array(self._id)
        N = bm.max(self.id) + 1
        self.cell_map = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.cell_map, self.id, bm.arange(len(self.id), dtype=bm.int32))
        for etype, conn_list in self._cell.items():
            # Convert each connectivity list to a bm.array
            self._cell[etype] = bm.array(conn_list)
        self.cell = self._cell

    def finalize_nastran(self, bdf) -> None:
        id_ = []
        cell_ = {}
        for elem_id, elem in bdf.elements.items():
            id_.append(elem_id)
            cell_.setdefault(mesh_type_map[elem.type], []).append(elem)
        for k, v in cell_.items():
            cell_[k] = bm.array([node.node_ids for node in v], dtype=bm.int64)
        self.id = bm.array(id_)
        N = bm.max(self.id) + 1
        self.cell_map = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.cell_map, self.id, bm.arange(len(self.id), dtype=bm.int32))
        for etype, conn_list in cell_.items():
            # Convert each connectivity list to a bm.array
            cell_[etype] = bm.array(conn_list)
        self.cell = cell_

    def attach(self, meshdata: Dict[str, Any]) -> None:
        meshdata['cell_map'] = self.cell_map  # 存入共享数据字典


# Registry of available section handlers
SECTION_REGISTRY: List[Type[Section]] = [NodeSection, ElementSection]