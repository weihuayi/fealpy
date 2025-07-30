
import re
from typing import List, Tuple, Dict, Type, Optional, Any
from ..backend import bm

class Section:
    """
    Abstract base class for all parsed ABAQUS *.inp file sections.

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

    def parse_line(self, line: str) -> None:
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
    Parses the *NODE section from an ABAQUS .inp file.

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

    def parse_line(self, line: str) -> None:
        parts = [s.strip() for s in line.split(',')]
        nid = int(parts[0])
        coords = [float(val) for val in parts[1:]]
        self._id.append(nid)
        self._node.append(coords)

    def finalize(self) -> None:
        self.id = bm.array(self._id)
        self.node = bm.array(self._node)
        N = bm.max(self.id) + 1
        self.node_map = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.node_map, self.id, bm.arange(len(self.id), dtype=bm.int32))

    def attach(self, meshdata: Dict[str, Any]) -> None:
        meshdata['node_map'] = self.node_map

class ElementSection(Section):
    """
    Parses the *ELEMENT section from an ABAQUS .inp file.

    This section defines element connectivity using global node IDs. It stores both raw
    data and post-processed tensor structures for efficient access.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header.

    Attributes:
        _id (List[int]): Element IDs parsed from each line.
        _cell (List[List[int]]): Element connectivity, where each list contains the node IDs of an element.
        id (bm.array | None): Array of element IDs.
        cell (bm.array | None): Array of element connectivities.
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
        self._cell: List[List[int]] = []
        # final arrays
        self.id: Optional[Any] = None
        self.cell: Optional[Any] = None
        self.cell_map: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = [s.strip() for s in line.split(',')]
        eid = int(parts[0])
        conn = [int(val) for val in parts[1:]]
        self._id.append(eid)
        self._cell.append(conn)

    def finalize(self) -> None:
        self.id = bm.array(self._id)
        N = bm.max(self.id) + 1
        self.cell_map = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.cell_map, self.id, bm.arange(len(self.id), dtype=bm.int32))
        self.cell = bm.array(self._cell)

    def attach(self, meshdata: Dict[str, Any]) -> None:
        meshdata['cell_map'] = self.cell_map  # 存入共享数据字典

class ElsetSection(Section):
    """
    Parses the *ELSET section from an ABAQUS .inp file.

    This section defines a named set of elements, which can be specified directly or generated via a range.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header, including 'elset' name and 'generate' flag.

    Attributes:
        name (str): Name of the element set.
        generate (bool): Whether the element IDs should be generated from a range.
        _id (List[int]): List of element IDs in the set.
        id (bm.array | None): Final array of element IDs.

    Methods:
        parse_line(line: str): Parses a line of element IDs or a generation command.
        finalize(): Converts internal storage to bm.array.
        attach(meshdata: Dict[str, Any]): Adds named element set to meshdata using 'cell_map'.
    """
    keyword = 'ELSET'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.name = options.get('elset', '')
        self.generate = options.get('generate', '').lower() == 'true'
        self._id: List[int] = []
        self.id: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = re.split(r'\s*,\s*', line)
        if self.generate:
            start, stop, step = map(int, parts[:3])
            self._id.extend(range(start, stop + 1, step))
        else:
            for p in parts:
                if p:
                    self._id.append(int(p))

    def finalize(self) -> None:
        self.id = bm.array(self._id)

    def attach(self, meshdata: Dict[str, Any]) -> None:
        if 'elset' not in meshdata:
            meshdata['elset'] = {}
        cell_map = meshdata['cell_map']
        meshdata['elset'][self.name] = cell_map[self.id]

class NsetSection(Section):
    """
    Parses the *NSET section from an ABAQUS .inp file.

    This section defines a named set of nodes, which are usually used to
    apply boundary conditions, loads, or group nodes for referencing in
    later sections such as materials or couplings.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header,
            including the 'nset' name.

    Attributes:
        name (str): Name of the node set.
        _id (List[int]): List of node IDs.
        id (bm.array | None): Final array of node indices.

    Methods:
        parse_line(line: str): Parses comma-separated node IDs from each line.
        finalize(): Converts node ID list to a tensor array.
        attach(meshdata: Dict[str, Any]): Adds the named node set to meshdata using the node map.
    """
    keyword = 'NSET'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.name = options.get('nset', '')
        self._id: List[int] = []
        self.id: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = re.split(r'\s*,\s*', line)
        for p in parts:
            if p:
                self._id.append(int(p))

    def finalize(self) -> None:
        self.id = bm.array(self._id)

    def attach(self, meshdata: Dict[str, Any]) -> None:
        if 'nset' not in meshdata:
            meshdata['nset'] = {}
        node_map = meshdata['node_map']
        meshdata['nset'][self.name] = node_map[self.id]


class SolidSection(Section):
    """
    Parses the *SOLID SECTION from an ABAQUS .inp file.

    This section maps a named element set to a specific material name. It
    typically contains no data lines but defines semantic groupings used
    for assigning materials in the mesh.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header,
            including 'elset' and 'material'.

    Attributes:
        elset (str): The name of the element set to which this solid section applies.
        material (str): The name of the material assigned to the element set.

    Methods:
        parse_line(line: str): No-op. Solid section information is header-only.
        attach(meshdata: Dict[str, Any]): Records the material-to-elset mapping in the meshdata dictionary.
    """
    keyword = 'SOLID SECTION'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.elset: Optional[str] = options.get('elset', '')
        self.material: Optional[str] = options.get('material', '')

    def parse_line(self, line: str) -> None:
        # Usually no data to parse; solid section info is in the header
        pass

    def attach(self, meshdata: Dict[str, Any]) -> None:
        if 'solid' not in meshdata:
            meshdata['solid'] = {}
        meshdata['solid'] = {
            'elset': self.elset,
            'material': self.material
        }


class SystemSection(Section):
    """
    Parses the *SYSTEM section from an ABAQUS .inp file.

    This section allows for storage of arbitrary system-level metadata or user-defined
    configuration content. It typically consists of plain text lines.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header.

    Attributes:
        data (List[str]): Raw lines of text data contained in the section.

    Methods:
        parse_line(line: str): Stores each non-empty line into the `data` list.
        attach(meshdata: Dict[str, Any]): Optional; can be overridden for post-processing.
    """
    keyword = 'SYSTEM'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.data = []

    def parse_line(self, line: str) -> None:
        self.data.append(line.strip())

class SurfaceSection(Section):
    """
    Parses the *SURFACE section from an ABAQUS .inp file.

    This section defines named surfaces, typically used for applying surface loads
    or for coupling and contact definitions. Each surface is defined by a set name
    and an associated type.

    Parameters:
        options (Dict[str, str]): Keyword arguments from the header line, including 'name' and 'type'.

    Attributes:
        name (str): Name of the surface.
        type (str): Surface type (e.g., 'ELEMENT', 'NODE', etc.).
        assignments (List[Tuple[str, float]]): List of surface assignments as (elset name, surface ID).

    Methods:
        parse_line(line: str): Parses each assignment line into (elset, surface ID).
        attach(meshdata: Dict[str, Any]): Adds the surface definition to meshdata.
    """
    keyword = 'SURFACE'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.name = options.get('name', '')
        self.type = options.get('type', '')
        self.assignments: List[Tuple[str, float]] = []

    def parse_line(self, line: str) -> None:
        parts = re.split(r'\s*,\s*', line.strip())
        if len(parts) >= 2:
            self.assignments.append((parts[0], float(parts[1])))

    def attach(self, meshdata: Dict[str, Any]):
        if 'surface' not in meshdata:
            meshdata['surface'] = {}
        meshdata['surface'][self.name] = {
            'type': self.type,
            'assignments': self.assignments,
        }


class CouplingSection(Section):
    """
    Parses the *COUPLING section from an ABAQUS .inp file.

    This section defines coupling constraints that relate a reference node to a surface.
    Additional lines such as *KINEMATIC or *DISTRIBUTING indicate the specific type of coupling.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header,
            including constraint name, reference node, and surface.

    Attributes:
        name (str): Name of the coupling constraint.
        ref_node (str): Reference node identifier.
        surface (str): Name of the coupled surface.
        type (str | None): Coupling type (e.g., 'KINEMATIC', 'DISTRIBUTING'), to be set externally.

    Methods:
        parse_line(line: str): No-op. All data is extracted from the header.
        set_type(coupling_type: str): Sets the type of coupling constraint.
        attach(meshdata: Dict[str, Any]): Adds coupling info to the meshdata dictionary.
    """
    keyword = 'COUPLING'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.name = options.get('constraint name', '')
        self.ref_node = options.get('ref node', '')
        self.surface = options.get('surface', '')
        self.type = None  # To be set by subsequent *KINEMATIC or *DISTRIBUTING

    def parse_line(self, line: str) -> None:
        pass  # No data expected; coupling info is in header

    def set_type(self, coupling_type: str) -> None:
        self.type = coupling_type

    def attach(self, meshdata: Dict[str, Any]):
        if 'coupling' not in meshdata:
            meshdata['coupling'] = {}
        meshdata['coupling'][self.name] = {
            'type': 'COUPLING',
            'ref_node': self.ref_node,
            'surface': self.surface,
            'coupling_type': self.type,
        }


class MaterialSection(Section):
    """
    Parses the *MATERIAL section from an ABAQUS .inp file.

    This section collects material properties such as density and elasticity, typically
    defined in sub-keyword lines like *DENSITY and *ELASTIC.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header,
            including the material name.

    Attributes:
        name (str): Name of the material.
        density (float | None): Material density.
        elastic (Tuple[float, float] | None): Young's modulus and Poisson's ratio.
        _next (str | None): Internal flag to indicate which property is expected next.

    Methods:
        parse_line(line: str): Handles parsing of property declarations and values.
        attach(meshdata: Dict[str, Any]): Adds the material data to meshdata under its name.
    """
    keyword = 'MATERIAL'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.name = options.get('name', '')
        self.density: Optional[float] = None
        self.elastic: Optional[Tuple[float, float]] = None
        self._next = None  # internal flag for line parsing

    def parse_line(self, line: str) -> None:
        if self._next == 'DENSITY':
            self.density = float(line.strip().split(',')[0])
            self._next = None
        elif self._next == 'ELASTIC':
            parts = [float(x) for x in line.strip().split(',') if x]
            if len(parts) >= 2:
                self.elastic = (parts[0], parts[1])
            self._next = None
        else:
            if line.upper().startswith("*DENSITY"):
                self._next = 'DENSITY'
            elif line.upper().startswith("*ELASTIC"):
                self._next = 'ELASTIC'

    def attach(self, meshdata: Dict[str, Any]):
        if 'material' not in meshdata:
            meshdata['material'] = {}
        meshdata['material'][self.name] = {
            'density': self.density,
            'elastic': self.elastic
        }


class BoundarySection(Section):
    """
    Parses the *BOUNDARY section from an ABAQUS .inp file.

    This section defines constraints on node degrees of freedom (DOFs), typically
    used for boundary conditions.

    Parameters:
        options (Dict[str, str]): Keyword arguments extracted from the section header.

    Attributes:
        boundaries (List[Tuple[str, int, int]]): List of boundary constraints in the format
            (node_set_name, dof_start, dof_end).

    Methods:
        parse_line(line: str): Parses a constraint definition line.
        attach(meshdata: Dict[str, Any]): Appends all boundary constraints to meshdata.
    """
    keyword = 'BOUNDARY'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.boundaries: List[Tuple[str, int, int]] = []

    def parse_line(self, line: str) -> None:
        parts = re.split(r'\s*,\s*', line.strip())
        if len(parts) >= 3:
            name = parts[0].upper()
            dof_start = int(parts[1])
            dof_end = int(parts[2])
            self.boundaries.append((name, dof_start, dof_end))

    def attach(self, meshdata: Dict[str, Any]):
        if 'boundary' not in meshdata:
            meshdata['boundary'] = []
        meshdata['boundary'].extend(self.boundaries)



# Registry of available section handlers
SECTION_REGISTRY: List[Type[Section]] = [NodeSection, ElementSection, ElsetSection, 
                                         NsetSection, SolidSection, SystemSection,
                                         SurfaceSection, CouplingSection, MaterialSection,
                                         BoundarySection]
