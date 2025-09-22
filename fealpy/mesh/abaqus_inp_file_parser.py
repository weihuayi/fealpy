import re
from typing import List, Dict, Type, Optional, Any
from ..backend import bm


class Section:
    """
    Base class for Abaqus .inp file sections. Subclasses should define `keyword`, implement `parse_line`, and optionally `finalize`.
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
        self._ids: List[int] = []
        self._coords_list: List[List[float]] = []
        # final arrays
        self.ids: Optional[Any] = None
        self.coords: Optional[Any] = None
        self.imap: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = [s.strip() for s in line.split(',')]
        nid = int(parts[0])
        coords = [float(val) for val in parts[1:]]
        self._ids.append(nid)
        self._coords_list.append(coords)

    def finalize(self) -> None:
        self.ids = bm.array(self._ids)
        self.coords = bm.array(self._coords_list)

        N = bm.max(self.ids) + 1
        self.imap = bm.zeros((N,), dtype=bm.int32)
        bm.set_at(self.imap, self.ids, bm.arange(len(self.ids), dtype=bm.int32))


class ElementSection(Section):
    """
    Parses the *Element section, storing element IDs and connectivity separately, then converts to bm.array.
    """
    keyword = 'ELEMENT'

    def __init__(self, options: Dict[str, str]):
        super().__init__(options)
        self._ids: List[int] = []
        self._conn_list: List[List[int]] = []
        # final arrays
        self.ids: Optional[Any] = None
        self.connectivity: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = [s.strip() for s in line.split(',')]
        eid = int(parts[0])
        conn = [int(val) for val in parts[1:]]
        self._ids.append(eid)
        self._conn_list.append(conn)

    def finalize(self) -> None:
        self.ids = bm.array(self._ids)
        self.connectivity = bm.array(self._conn_list)


# Registry of available section handlers
SECTION_REGISTRY: List[Type[Section]] = [NodeSection, ElementSection]


class AbaqusInpFileParser:
    """
    Main parser class for Abaqus .inp files. Identifies sections, delegates parsing, and finalizes arrays.
    """
    def __init__(self) -> None:
        self.sections: List[Section] = []

    def parse(self, filename: str) -> 'AbaqusInpParser':
        current_section: Optional[Section] = None
        with open(filename, 'r') as f:
            for raw in f:
                line = raw.strip()
                # skip empty and comment lines
                if not line or line.startswith('**'):
                    continue
                if line.startswith('*'):
                    # start a new section
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
            if '=' in part:
                k, v = part.split('=', 1)
                options[k.strip()] = v.strip()
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

    def to_mesh(self, mesh_type):
        nodes = self.get_section(NodeSection)
        elems = self.get_section(ElementSection)
        node = nodes.coords
        cell = nodes.imap[elems.connectivity]
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

