import re
from typing import List, Tuple, Dict, Type, Optional, Any
from ..backend import bm


class MSection:
    """M 文件段落基类"""
    keyword: str = ''

    @classmethod
    def match_keyword(cls, token: str) -> bool:
        return cls.keyword == token

    def parse_line(self, line: str) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        pass


class VertexSection(MSection):
    """
    解析 Vertex 行。
    严格格式示例:
    Vertex 31 165.9 164.8 35.6 {uv=(0.78 0.60) rgb=(0.12 0.12 0.12)}
    """
    keyword = 'Vertex'

    def __init__(self):
        self._ids: List[int] = []
        self._coords: List[List[float]] = []
        self._uvs: List[List[float]] = []
        self._rgbs: List[List[float]] = []

        # 编译正则以提高性能
        # 匹配 uv=(...)
        self.uv_pattern = re.compile(r"uv=\(([\d\.\seE+-]+)\)")
        # 匹配 rgb=(...)
        self.rgb_pattern = re.compile(r"rgb=\(([\d\.\seE+-]+)\)")

        # 结果存储
        self.node: Optional[Any] = None
        self.uv: Optional[Any] = None
        self.rgb: Optional[Any] = None
        self.id_map: Dict[int, int] = {}  # 原始ID -> 内部索引

    def parse_line(self, line: str) -> None:
        parts = line.split()
        try:
            # 1. 基础坐标: Vertex <id> <x> <y> <z>
            vid = int(parts[1])
            coords = [float(x) for x in parts[2:5]]

            self._ids.append(vid)
            self._coords.append(coords)

            # 2. 解析大括号 {} 内的属性
            # 找到大括号内容
            brace_start = line.find('{')
            brace_end = line.rfind('}')

            if brace_start != -1 and brace_end != -1:
                props_str = line[brace_start + 1:brace_end]

                # 提取 UV
                uv_match = self.uv_pattern.search(props_str)
                if uv_match:
                    self._uvs.append([float(x) for x in uv_match.group(1).split()])
                else:
                    self._uvs.append([0.0, 0.0])  # 默认值

                # 提取 RGB
                rgb_match = self.rgb_pattern.search(props_str)
                if rgb_match:
                    self._rgbs.append([float(x) for x in rgb_match.group(1).split()])
                else:
                    self._rgbs.append([0.0, 0.0, 0.0])  # 默认值
            else:
                # 没有任何属性的情况
                self._uvs.append([0.0, 0.0])
                self._rgbs.append([0.0, 0.0, 0.0])

        except (ValueError, IndexError):
            pass

    def finalize(self) -> None:
        self.node = bm.array(self._coords, dtype=bm.float64)
        if self._uvs:
            self.uv = bm.array(self._uvs, dtype=bm.float64)
        if self._rgbs:
            self.rgb = bm.array(self._rgbs, dtype=bm.float64)

        # 构建 ID 映射表: 原始文件ID -> 数组下标 0, 1, 2...
        self.id_map = {vid: i for i, vid in enumerate(self._ids)}


class FaceSection(MSection):
    """
    解析 Face 行。
    格式: Face <fid> <v1> <v2> <v3> ...
    """
    keyword = 'Face'

    def __init__(self):
        self._raw_faces: List[List[int]] = []  # 存储原始 ID
        self.cell: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = line.split()
        try:
            # parts[0]=Face, parts[1]=FaceID
            # 从 parts[2] 开始是顶点 ID
            v_indices = []
            for token in parts[2:]:
                # 遇到非数字字符（如 {属性}）停止
                if not token.lstrip('-').isdigit():
                    break
                v_indices.append(int(token))

            if len(v_indices) >= 3:
                self._raw_faces.append(v_indices)
        except ValueError:
            pass

    def process_cells(self, id_map: Dict[int, int]) -> Any:
        """利用 id_map 将原始 ID 转为索引并三角化"""
        triangles = []
        for f_vids in self._raw_faces:
            # 映射 ID -> Index
            # 如果某个点不在 vertex 列表里（极其少见），则跳过
            mapped = [id_map[vid] for vid in f_vids if vid in id_map]

            if len(mapped) < 3: continue

            # Fan Triangulation (v0, v1, v2), (v0, v2, v3)...
            v0 = mapped[0]
            for i in range(1, len(mapped) - 1):
                triangles.append([v0, mapped[i], mapped[i + 1]])

        self.cell = bm.array(triangles, dtype=bm.int64)
        return self.cell


class EdgeSection(MSection):
    """
    解析 Edge 行。
    格式: Edge <v1> <v2> {sharp}
    注意：示例中 Edge 似乎没有 ID，只有两个端点和属性
    """
    keyword = 'Edge'

    def __init__(self):
        self._raw_edges: List[Tuple[int, int]] = []
        self._is_sharp: List[bool] = []

        self.edge: Optional[Any] = None
        self.sharp_mask: Optional[Any] = None

    def parse_line(self, line: str) -> None:
        parts = line.split()
        try:
            # 格式: Edge v1 v2 {sharp}
            # parts[0]=Edge
            v1 = int(parts[1])
            v2 = int(parts[2])

            self._raw_edges.append((v1, v2))

            # 检查是否包含 {sharp} 标记
            if "{sharp}" in line:
                self._is_sharp.append(True)
            else:
                self._is_sharp.append(False)
        except (ValueError, IndexError):
            pass

    def process_edges(self, id_map: Dict[int, int]) -> Tuple[Any, Any]:
        """处理边数据，映射 ID，返回 (edges, sharp_flags)"""
        valid_edges = []
        valid_flags = []

        for (v1, v2), is_s in zip(self._raw_edges, self._is_sharp):
            if v1 in id_map and v2 in id_map:
                valid_edges.append([id_map[v1], id_map[v2]])
                valid_flags.append(is_s)

        if valid_edges:
            self.edge = bm.array(valid_edges, dtype=bm.int64)
            self.sharp_mask = bm.array(valid_flags, dtype=bool)
        else:
            # 如果没有读取到边，返回空
            self.edge = bm.zeros((0, 2), dtype=bm.int64)
            self.sharp_mask = bm.zeros((0,), dtype=bool)

        return self.edge, self.sharp_mask


# 注册所有 Section
M_SECTION_REGISTRY: List[Type[MSection]] = [VertexSection, FaceSection, EdgeSection]