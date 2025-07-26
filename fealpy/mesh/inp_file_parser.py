import re
from typing import List, Tuple, Dict, Type, Optional, Any
from ..backend import bm


class Section:
    """
    Base class for .inp file sections.

    Attributes
        options : Dict[str, str]
            Stores the parsed keyword arguments provided to this section.
        keyword : str
            The keyword name associated with the section. Subclasses must define this.
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
    Parses the *Node section, storing node IDs and coordinates separately, then converts to `bm.array`.

    Parameters
        options : Dict[str, str]
            Keyword options parsed from the section header line, passed from the base class.

    Attributes
        _id : List[int]
            List of node IDs parsed from the file.
        _node : List[List[float]]
            List of node coordinates, each as a list of floats.
        id : bm.array, optional
            Array of node IDs.
        node : bm.array, optional
            Array of node coordinates with shape (num_nodes, dim).
        imap : bm.array, optional
            Index mapping from node ID to row index in the node array. Shape: (max_id + 1,)
    """
    keyword = 'NODE'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self._id: List[int] = []
        self._node: List[List[float]] = []
        # final arrays
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
        meshdata['node_map'] = self.node_map  # 存入共享数据字典


class ElementSection(Section):
    """Parses the *Element section, storing element (cell) IDs and connectivity separately, 
    then converts to `bm.array`.

    Parameters
        options : Dict[str, str]
            Keyword options parsed from the section header line, passed from the base class.

    Attributes
        _id : List[int]
            List of element (cell) IDs parsed from the file.
        _cell : List[List[int]]
            List of element connectivity, where each entry contains node indices of the element.
        id : bm.array, optional
            Array of element IDs.
        cell : bm.array, optional
            Array of element connectivities. Shape: (num_elements, num_nodes_per_element)
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
    Parses the *Elset section, storing named element sets.
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
            # e.g., 1, 10, 1  means: 1 2 3 ... 10
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
    Parses the *Nset section, storing named node sets.
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
    Parses the *Solid Section definition, extracting elset and material.
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
    keyword = 'SYSTEM'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self.data = []

    def parse_line(self, line: str) -> None:
        self.data.append(line.strip())


class SurfaceSection(Section):
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


class InpFileParser:
    """
    Main parser class for inp files. Identifies sections, delegates parsing, and finalizes arrays.
    """
    def __init__(self) -> None:
        self.sections: List[Section] = []

    def parse(self, filename: str) -> 'InpFileParser':
        current_section: Optional[Section] = None
        with open(filename, 'r') as f:
            for raw in f:
                line = raw.strip()
                # skip empty and comment lines
                if not line or line.startswith('**'):
                    continue
                if line.startswith('*'):
                    # start a new section
                    keyword = line[1:].split(',')[0].strip()
                    if keyword in ('Kinematic', 'distributing'):
                        if isinstance(current_section, CouplingSection):
                            current_section.set_type(keyword)
                        continue
                    if isinstance(current_section, MaterialSection) and keyword.upper() in ('DENSITY', 'ELASTIC'):
                        # 交由 MaterialSection 内部处理
                        current_section.parse_line(line)
                        continue
                    # 否则是新的 Section
                    current_section = self._start_section(line)
                    if current_section:
                        self.sections.append(current_section)
                    continue
                # parse data lines
                if current_section:
                    current_section.parse_line(line)
        # finalize all sections (convert to bm.array)
        for sec in self.sections:
            sec.finalize()
        return self

    def _start_section(self, header: str) -> Optional[Section]:
        # remove leading '*' and split by commas
        parts = re.split(r"\s*,\s*", header[1:])
        keyword = parts[0]
        options: Dict[str, str] = {}
        for part in parts[1:]:
            if not part.strip():
                continue  # 跳过空字符串
            if '=' in part:
                k, v = part.split('=', 1)
                options[k.strip()] = v.strip()
            else:
                options[part.strip().lower()] = 'true'
        # find matching section handler
        for sec_cls in SECTION_REGISTRY:
            if sec_cls.match_keyword(keyword):
                return sec_cls(options)
        return None

    def get_section(self, section_type: Type[Section]) -> Optional[Section]:
        for sec in self.sections:
            if isinstance(sec, section_type):
                return sec
        return None

    def get_sections(self, section_type: Type[Section]) -> List[Section]:
        return [sec for sec in self.sections if isinstance(sec, section_type)]
  
    def to_mesh(self, mesh_type):
        ns = self.get_section(NodeSection)
        es = self.get_section(ElementSection)
        node = ns.node
        cell = ns.node_map[es.cell]
        mesh = mesh_type(node, cell)

        for section in self.sections:
            section.attach(mesh.meshdata)
        return mesh

    def to_material(self, Material, name: str):
        materials = self.get_section(MaterialSection)
        elastic_modulus, poisson_ratio = materials.elastic
        density = materials.density

        return Material(
            name=name,
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            density=density
        )

# Example usage:
if __name__ == '__main__':
    parser = InpFileParser()
    parser.parse('/home/why/fealpy/data/LANXIANG_KETI_0506.inp')
    nodes = parser.get_section(NodeSection)
    elems = parser.get_section(ElementSection)
    if nodes and elems:
        print(f"Read {nodes.ids.shape[0]} nodes and {elems.ids.shape[0]} elements.")
    else:
        print("Required sections not found in the inp file.")

