from typing import Tuple
from math import ceil, sqrt

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor
from fealpy.mesh import UniformMesh2d, UniformMesh3d

class FilterMatrix:
    """滤波矩阵计算工具类"""
    
    @staticmethod
    def create_filter_matrix(mesh, filter_radius: float) -> Tuple[COOTensor, TensorLike]:
        """
        创建网格依赖滤波矩阵 (仅支持单位长度的均匀网格)
        
        Parameters
        - mesh : 网格
        - radius : 滤波半径
            
        Returns
        - H : 滤波矩阵
        - Hs : 滤波矩阵行和向量
        """
        if isinstance(mesh, UniformMesh2d):
            if not (mesh.h[0] == mesh.h[1] == 1.0):
                raise ValueError("FilterMatrix only supports uniform mesh with unit length (h[0] = h[1] = 1.0)")
            return FilterMatrix._compute_filter_2d(mesh.nx, mesh.ny, filter_radius)
        elif isinstance(mesh, UniformMesh3d):
            if not (mesh.h[0] == mesh.h[1] == mesh.h[2] == 1.0):
                raise ValueError("FilterMatrix only supports uniform mesh with unit length (h[0] = h[1] = h[2] = 1.0)")
            return FilterMatrix._compute_filter_3d(mesh.nx, mesh.ny, mesh.nz, filter_radius)
        else:
            raise TypeError("Mesh must be UniformMesh2d or UniformMesh3d")
    
    @staticmethod
    def _compute_filter_2d(nx: int, ny: int, rmin: float) -> Tuple[COOTensor, TensorLike]:
        """计算 UniformMesh2d 下的滤波矩阵"""
        
        nfilter = int(nx * ny * ((2 * (ceil(rmin) - 1) + 1) ** 2))
        iH = bm.zeros(nfilter, dtype=bm.int32)
        jH = bm.zeros(nfilter, dtype=bm.int32)
        sH = bm.ones(nfilter, dtype=bm.float64)
        cc = 0

        for i in range(nx):
            for j in range(ny):
                # 单元的编号顺序: y->x 
                row = i * ny + j
                kk1 = int(max(i - (ceil(rmin) - 1), 0))
                kk2 = int(min(i + ceil(rmin), nx))
                ll1 = int(max(j - (ceil(rmin) - 1), 0))
                ll2 = int(min(j + ceil(rmin), ny))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * ny + l
                        fac = rmin - sqrt((i - k) ** 2 + (j - l) ** 2)
                        if fac > 0:
                            iH[cc] = row
                            jH[cc] = col
                            sH[cc] = max(0.0, fac)
                            # sH[cc] = fac
                            cc += 1

        H = COOTensor(
            indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
            values=sH[:cc],
            spshape=(nx * ny, nx * ny)
        )
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)
        
        return H, Hs

    @staticmethod
    def _compute_filter_3d(nx: int, ny: int, nz: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
        """计算 3D 滤波矩阵"""

        ceil_rmin = int(ceil(rmin))
        nfilter = nx * ny * nz * ((2 * (ceil_rmin - 1) + 1) ** 3)
        iH = bm.zeros(nfilter, dtype=bm.int32)
        jH = bm.zeros(nfilter, dtype=bm.int32)
        sH = bm.zeros(nfilter, dtype=bm.float64)
        cc = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 单元的编号顺序: z -> y -> x
                    row = k + j * nz + i * ny * nz 
                    ii1 = max(i - (ceil_rmin - 1), 0)
                    ii2 = min(i + (ceil_rmin - 1), nx - 1)
                    jj1 = max(j - (ceil_rmin - 1), 0)
                    jj2 = min(j + (ceil_rmin - 1), ny - 1)
                    kk1 = max(k - (ceil_rmin - 1), 0)
                    kk2 = min(k + (ceil_rmin - 1), nz - 1)
                    for ii in range(ii1, ii2 + 1):
                        for jj in range(jj1, jj2 + 1):
                            for kk in range(kk1, kk2 + 1):
                                # 单元的编号顺序: z -> y -> x
                                col = kk + jj * nz + ii * ny * nz 
                                fac = rmin - sqrt((i - ii)**2 + (j - jj)**2 + (k - kk)**2)
                                iH[cc] = row
                                jH[cc] = col
                                sH[cc] = max(0.0, fac)
                                cc += 1

        H = COOTensor(indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32), 
                      values=sH[:cc], 
                      spshape=(nx * ny * nz, nx * ny * nz))
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)

        return H, Hs