import re
from typing import List, Tuple, Dict, Type, Optional, Any
from fealpy.backend import bm
from .bdf_file_sections import *


class BdfFileParser:
    """
    A parser for Nastran .bdf files, supporting both free and fixed formats.

    This class parses nodes (GRID cards) and elements (CTRIA3, CQUAD4, CTETRA, CHEXA cards)
    from a .bdf file, storing nodes in a array and elements in a dictionary keyed by element type.
    The parsed content can be used to construct mesh objects suitable for FEALPy.

    Attributes:
        sections (List[Section]): A list of parsed section instances in order of appearance.
        is_free_format (bool): Indicates whether the file is in free format (,-separated) or fixed format.

    Methods:
        parse(filename: str) -> BdfFileParser:
            Parses the given .bdf file and returns the parser instance itself.

        get_section(section_type: Type[Section]) -> Optional[Section]:
            Returns the first instance of a given section type (e.g., NodeSection).

        get_sections(section_type: Type[Section]) -> List[Section]:
            Returns all instances of a given section type.

        to_mesh(mesh_type):
            Constructs a mesh instance using the parsed node and element data,
            when there is only one element section.

    """

    def __init__(self) -> None:
        self.sections: List[Section] = []
        self.is_free_format = None

    def parse(self, filename: str, is_use_nastran:bool=False) -> 'BdfFileParser':
        current_section: Optional[Section] = None
        if is_use_nastran:
            try:
                from pyNastran.bdf.bdf import BDF
            except ImportError:
                raise ImportError("pyNastran is not installed. Please install it to use BDF file parsing.")

            nastran_card_map = {'GRID': 'NODE',
                                'CTRIA3': 'ELEMENT',
                                'CQUAD4': 'ELEMENT',
                                'CTETRA': 'ELEMENT',
                                'CHEXA': 'ELEMENT'}
            # 初始化 BDF 对象
            bdf = BDF()
            try:
                bdf.read_bdf(filename, punch=False)
            except Exception as e:
                try:
                    # 如果读取失败，尝试不使用 punch
                    bdf.read_bdf(filename, punch=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to read BDF file: {filename}, Error: {e}. "
                                       f"try to delete the fist line with 'BEGIN BULK' "
                                       f"and the last line with 'ENDDATA' in the file.")
            for card in bdf.card_count.keys():
                is_exist_section = False
                if card in nastran_card_map:
                    options: Dict[str, str] = {}
                    for sec in self.sections:
                        if sec.match_keyword(nastran_card_map[card]):
                            is_exist_section = True
                            break
                    if not is_exist_section:
                        for sec_cls in SECTION_REGISTRY:
                            if sec_cls.match_keyword(nastran_card_map[card]):
                                current_section = sec_cls(options)
                                self.sections.append(current_section)
            for sec in self.sections:
                sec.finalize_nastran(bdf)
            return self

        with open(filename, 'r') as f:
            for raw in f:
                line = raw.strip()
                # skip empty and comment lines
                if not line or line.startswith('$'):
                    continue
                if line.startswith('GRID') \
                        or line.startswith('CTRIA3') or line.startswith('CQUAD4')\
                        or line.startswith('CTETRA') or line.startswith('CHEXA'):
                    if self.is_free_format is None:
                        self.is_free_format = ',' in line and not line.startswith('$')
                    if line.startswith('GRID'):
                        keyword = 'NODE'
                    else:
                        keyword = 'ELEMENT'
                    # 否则是新的 Section
                    options: Dict[str, str] = {}
                    is_exist_section = False
                    for sec in self.sections:
                        if sec.match_keyword(keyword):
                            current_section = sec
                            is_exist_section = True
                            break
                    if not is_exist_section:
                        for sec_cls in SECTION_REGISTRY:
                            if sec_cls.match_keyword(keyword):
                                current_section = sec_cls(options)
                                self.sections.append(current_section)
                    current_section.parse_line(line, is_free_format=self.is_free_format)
                    continue
        # finalize all sections (convert to bm.array)
        for sec in self.sections:
            sec.finalize()
        return self

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
        if len(es) == 1:
            es = es[0]
        else:
            raise ValueError("Multiple element sections found, cannot determine mesh type.")
        node = ns.node
        cell = ns.node_map[es.cell]
        mesh = mesh_type(node, cell)

        for section in self.sections:
            section.attach(mesh.meshdata)
        return mesh


