import re
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


# Registry of available section handlers
SECTION_REGISTRY: List[Type[Section]] = [NodeSection, ElementSection, ElsetSection, NsetSection]


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
        cell = ns.imap[es.cell]
        return mesh_type(node, cell) 

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

