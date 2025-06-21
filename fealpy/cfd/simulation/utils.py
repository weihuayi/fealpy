from fealpy.backend import backend_manager as bm
import numpy as np
import pyvista
from typing import Dict

class VTKWriter:
    def __init__(self):
        pass

    def write_vtk(self, data_dict: Dict, path: str):
        """Store a .vtk file for ParaView."""
        data_pv = self.dict2pyvista(data_dict)
        data_pv.save(path)

    def dict2pyvista(self, data_dict: Dict) -> pyvista.PolyData:
        r = np.asarray(data_dict["position"])
        N, dim = r.shape

        # 如果是2D数据，自动升为3D
        if dim == 2:
            r = np.hstack([r, np.zeros((N, 1))])
        data_pv = pyvista.PolyData(r)

        for k, v in data_dict.items():
            if k == "position":
                continue
            v = np.asarray(v)
            if v.ndim == 0:
                # 标量，复制成 (N,) 形式
                v = np.full((N,), v)
                data_pv[k] = v
            elif v.ndim == 1:
                # 一维向量 (N,)
                data_pv[k] = v
            elif v.ndim == 2:
                # 二维向量或矩阵 (N, dim)
                if dim == 2 and v.shape[1] == 2:
                    v = np.hstack([v, np.zeros((N, 1))])
                data_pv[k] = v
            else:
                raise ValueError(f"Unsupported data shape for key '{k}': {v.shape}")
        return data_pv


