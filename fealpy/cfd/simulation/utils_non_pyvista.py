import numpy as np
import vtk
from typing import Dict
from pathlib import Path

class VTKWriter2:
    def __init__(self):
        pass

    def write_vtk(self, data_dict: Dict, path):
        path = Path(path)  # 接收 str 或 Path，都统一成 Path

        poly = self.dict2vtk(data_dict)

        # 根据后缀判断格式
        suf = path.suffix.lower()
        if suf == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        elif suf == ".vtk":
            writer = vtk.vtkPolyDataWriter()
        else:
            raise ValueError(f"Unsupported extension {suf}, use .vtk or .vtp")

        # 自动创建目录
        path.parent.mkdir(parents=True, exist_ok=True)

        writer.SetFileName(str(path))  # VTK 要求字符串
        writer.SetInputData(poly)
        writer.Write()
    
    def dict2vtk(self, data_dict: Dict) -> vtk.vtkPolyData:
        r = np.asarray(data_dict["position"])
        N, dim = r.shape

        # 升维 2D → 3D
        if dim == 2:
            r = np.hstack([r, np.zeros((N, 1))])

        # VTK points
        points = vtk.vtkPoints()
        for p in r:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)

        # 添加每一个字段作为点数据
        for key, value in data_dict.items():
            if key == "position":
                continue

            v = np.asarray(value)

            # 标量 → (N,)
            if v.ndim == 0:
                v = np.full((N,), float(v))

            # 一维 → 标量
            if v.ndim == 1:
                arr = vtk.vtkDoubleArray()
                arr.SetName(key)
                arr.SetNumberOfComponents(1)
                arr.SetNumberOfTuples(N)
                for i in range(N):
                    arr.SetValue(i, float(v[i]))
                poly.GetPointData().AddArray(arr)

            # 二维 → 向量/张量
            elif v.ndim == 2:
                comp = v.shape[1]
                # 升维到 3D（与位置保持一致）
                if comp == 2:
                    v = np.hstack([v, np.zeros((N, 1))])
                    comp = 3

                arr = vtk.vtkDoubleArray()
                arr.SetName(key)
                arr.SetNumberOfComponents(comp)
                arr.SetNumberOfTuples(N)

                for i in range(N):
                    arr.SetTuple(i, v[i].tolist())

                poly.GetPointData().AddArray(arr)

            else:
                raise ValueError(f"Unsupported shape for key '{key}': v.shape={v.shape}")

        return poly
