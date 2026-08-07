import re
from typing import List, Optional, Type, Any
from ..backend import bm
from .m_file_sections import (MSection, M_SECTION_REGISTRY,
                              VertexSection, FaceSection,
                              EdgeSection)


class MFileParser:
    def __init__(self) -> None:
        self.sections: List[MSection] = []

    def parse(self, filename: str) -> 'MFileParser':
        current_section: Optional[MSection] = None
        current_keyword: str = ""

        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue

                # 获取行首单词
                parts = stripped_line.split(maxsplit=1)
                keyword = parts[0]

                # 切换 Section 逻辑
                if keyword != current_keyword:
                    # 尝试匹配新 Section
                    new_section = self._match_section(keyword)
                    if new_section:
                        current_section = new_section
                        current_keyword = keyword
                        self.sections.append(current_section)
                    # 如果匹配不到，保持当前 section (容错) 或跳过

                # 解析行
                if current_section:
                    current_section.parse_line(stripped_line)

        # 基础数据转换 (如 Vertex 中的 coordinates)
        for sec in self.sections:
            sec.finalize()

        return self

    def _match_section(self, keyword: str) -> Optional[MSection]:
        for sec_cls in M_SECTION_REGISTRY:
            if sec_cls.match_keyword(keyword):
                return sec_cls()
        return None

    def get_section(self, section_type: Type[MSection]) -> Optional[MSection]:
        for sec in self.sections:
            if isinstance(sec, section_type):
                return sec
        return None

    def to_mesh(self, mesh_type, meshdata_type=None):
        """
        组装 mesh 对象
        :param mesh_type: 如 TriangleMesh
        :param meshdata_type: 可选的 MeshData 类，用于挂载额外属性
        """
        # 1. 处理节点 (Vertex)
        v_sec = self.get_section(VertexSection)
        if v_sec is None:
            raise ValueError("M file must contain Vertex data.")

        node = v_sec.node

        # 2. 处理单元 (Face) - 需要 ID 映射
        f_sec = self.get_section(FaceSection)
        if f_sec is None:
            raise ValueError("M file must contain Face data.")

        cell = f_sec.process_cells(v_sec.id_map)

        # 3. 初始化 Mesh
        mesh = mesh_type(node, cell)

        # 4. 挂载数据
        # 4.1 挂载节点属性 (uv, rgb)
        if v_sec.uv is not None:
            mesh.nodedata['uv'] = v_sec.uv
        if v_sec.rgb is not None:
            mesh.nodedata['rgb'] = v_sec.rgb  # 或者 'color'

        # 4.2 处理特征边 (Edge)
        e_sec = self.get_section(EdgeSection)
        if e_sec:
            # 处理边并获取 sharp 标记
            feature_edges, sharp_flags = e_sec.process_edges(v_sec.id_map)

            if len(feature_edges) > 0:
                pass
                # # 将读取到的边存储在 edgedata 中
                # # 注意：fealpy 的 mesh.ds.edge 通常是根据 cell 自动生成的
                # # 这里读取的是“显式定义的边”，通常是特征边或切割线，所以名字区分开
                # mesh.edgedata['feature_edge'] = feature_edges
                # mesh.edgedata['sharp'] = sharp_flags

        return mesh