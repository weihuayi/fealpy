import re
<<<<<<< HEAD
from typing import List, Dict, Type, Optional, Any
from ..backend import bm


class Section:
    """
    Base class for inp file sections. Subclasses should define `keyword`, implement `parse_line`, and optionally `finalize`.
    """
    keyword: str = ''

    def __init__(self, options: Dict[str, str]):
        self.options = options

    @classmethod
    def match_keyword(cls, keyword: str) -> bool:
        return cls.keyword.upper() == keyword.upper()

    def parse_line(self, line: str) -> None:
        raise NotImplementedError(f"parse_line must be implemented by {self.__class__.__name__}")

    def finalize(self) -> None:
        """
        Optional: convert stored lists to bm.array
        """
        pass


class NodeSection(Section):
    """
    Parses the *Node section, storing node IDs and coordinates separately, then converts to bm.array.
    """
    keyword = 'NODE'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self._id: List[int] = []
        self._node: List[List[float]] = []
        # final arrays
        self.id: Optional[Any] = None
        self.node: Optional[Any] = None
        self.imap: Optional[Any] = None

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
        self.imap = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.imap, self.id, bm.arange(len(self.id), dtype=bm.int32))


class ElementSection(Section):
    """
    Parses the *Element section, storing element(cell) IDs and connectivity separately, then converts to bm.array.
    """
    keyword = 'ELEMENT'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self._id: List[int] = []
        self._cell: List[List[int]] = []
        # final arrays
        self.id: Optional[Any] = None
        self.cell: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = [s.strip() for s in line.split(',')]
        eid = int(parts[0])
        conn = [int(val) for val in parts[1:]]
        self._id.append(eid)
        self._cell.append(conn)

    def finalize(self) -> None:
        self.id = bm.array(self._id)
        self.cell = bm.array(self._cell)


# Registry of available section handlers
SECTION_REGISTRY: List[Type[Section]] = [NodeSection, ElementSection]
=======
from typing import List, Tuple, Dict, Type, Optional, Any
from ..backend import bm
from ..typing import TensorLike
from .inp_file_sections import *
>>>>>>> origin/develop


class InpFileParser:
    """
<<<<<<< HEAD
    Main parser class for inp files. Identifies sections, delegates parsing, and finalizes arrays.
=======
    Main parser class for ABAQUS .inp files.

    This class reads and interprets .inp files, which define finite element models.
    It identifies section headers (e.g., *NODE, *ELEMENT), delegates parsing to the
    corresponding `Section` subclass, and postprocesses each section. The parsed
    content can be used to construct mesh and material objects suitable for FEALPy.

    Attributes:
        sections (List[Section]): A list of parsed section instances in order of appearance.

    Methods:
        parse(filename: str) -> InpFileParser:
            Parses the given .inp file and returns the parser instance itself.

        _start_section(header: str) -> Optional[Section]:
            Internal helper to match a section keyword with its corresponding handler class
            and create the section instance.

        get_section(section_type: Type[Section]) -> Optional[Section]:
            Returns the first instance of a given section type (e.g., NodeSection).

        get_sections(section_type: Type[Section]) -> List[Section]:
            Returns all instances of a given section type.

        to_mesh(mesh_type):
            Constructs a mesh instance using the parsed node and element data.
            Automatically attaches other section data to `mesh.meshdata`.

        to_material(Material, name: str):
            Constructs a material instance using parsed material parameters.
>>>>>>> origin/develop
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
<<<<<<< HEAD
=======
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
>>>>>>> origin/develop
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
<<<<<<< HEAD
            if '=' in part:
                k, v = part.split('=', 1)
                options[k.strip()] = v.strip()
=======
            if not part.strip():
                continue  # 跳过空字符串
            if '=' in part:
                k, v = part.split('=', 1)
                options[k.strip()] = v.strip()
            else:
                options[part.strip().lower()] = 'true'
>>>>>>> origin/develop
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

<<<<<<< HEAD
    def to_mesh(self, mesh_type):
        ns = self.get_section(NodeSection)
        es = self.get_section(ElementSection)
        node = ns.node
        cell = ns.imap[es.cell]
        return mesh_type(node, cell) 

# Example usage:
if __name__ == '__main__':
    parser = AbaqusInpParser()
    parser.parse('/home/why/fealpy/data/LANXIANG_KETI_0506.inp')
    nodes = parser.get_section(NodeSection)
    elems = parser.get_section(ElementSection)
    if nodes and elems:
        print(f"Read {nodes.ids.shape[0]} nodes and {elems.ids.shape[0]} elements.")
    else:
        print("Required sections not found in the inp file.")

=======
    def get_sections(self, section_type: Type[Section]) -> List[Section]:
        return [sec for sec in self.sections if isinstance(sec, section_type)]
  
    def to_mesh(self, mesh_type, meshdata_type):
        """
        """
        meshdata  = meshdata_type() 
        for section in self.sections:
            section.attach(meshdata)

        ns = self.get_section(NodeSection)
        es = self.get_section(ElementSection)
        node = ns.node
        cell = meshdata.update_node_id(es.cell)
        mesh = mesh_type(node, cell)
        mesh.data = meshdata 

        return mesh
>>>>>>> origin/develop
