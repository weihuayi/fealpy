import re
from typing import List, Tuple, Dict, Type, Optional, Any
from ..backend import bm
from ..typing import TensorLike
from .inp_file_sections import *


class InpFileParser:
    """
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
    """
    def __init__(self) -> None:
        self.sections: List[Section] = []
        self.nsets: Dict[str, TensorLike] = {}
        self.esets: Dict[str, TensorLike] = {}
        self.surfaces: Dict[str, SurfaceSection] = {}
        self.couplings: Dict[str, CouplingSection] = {}

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
            if isinstance(sec, NsetSection):
                name = sec.name
                self.nsets[name] = sec.id
            elif isinstance(sec, ElsetSection): 
                name = sec.name
                self.esets[name] = sec.id
            elif isinstance(sec, SurfaceSection):
                name = sec.name
                self.surfaces[name] = sec
            elif isinstance(sec, CouplingSection):
                name = sec.name
                self.couplings[name] = sec

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
        cell = ns.id_map[es.cell]
        mesh = mesh_type(node, cell)

        for section in self.sections:
            section.attach(mesh.meshdata, self)

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
