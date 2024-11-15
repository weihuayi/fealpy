from math import ceil, sqrt
from typing import Tuple
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor
from fealpy.backend import backend_manager as bm
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.mesh.uniform_mesh_3d import UniformMesh3d
from fealpy.mesh.mesh_base import Mesh

class FilterMatrixBuilder:
    """滤波矩阵构建器"""
    
    @staticmethod
    def build(mesh: Mesh, rmin: float) -> Tuple[TensorLike, TensorLike]:
        """构建滤波矩阵
        
        Args:
            mesh: 网格对象
            rmin: 过滤半径
            
        Returns:
            Tuple[COOTensor, TensorLike]: 过滤矩阵和缩放向量
            
        Raises:
            TypeError: 当网格类型不支持时
            ValueError: 当参数无效时
        """
        if not isinstance(mesh, (UniformMesh2d, UniformMesh3d)):
            raise TypeError("mesh must be an instance of UniformMesh2d or UniformMesh3d")
            
        if rmin <= 0:
            raise ValueError("Filter radius (rmin) must be positive")
            
        builder = FilterMatrixBuilder()
        if isinstance(mesh, UniformMesh2d):
            return builder._build_2d(mesh.nx, mesh.ny, rmin)
        else:
            return builder._build_3d(mesh.nx, mesh.ny, mesh.nz, rmin)
    
    def _build_2d(self, nx: int, ny: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
        """构建2D滤波矩阵"""
        nfilter = int(nx * ny * ((2 * (ceil(rmin) - 1) + 1) ** 2))
        iH = bm.zeros(nfilter, dtype=bm.int32)
        jH = bm.zeros(nfilter, dtype=bm.int32)
        sH = bm.ones(nfilter, dtype=bm.float64)
        cc = 0
        
        for i in range(nx):
            for j in range(ny):
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
                            sH[cc] = fac
                            cc += 1
        
        return self._create_matrix(iH[:cc], jH[:cc], sH[:cc], nx * ny)
    
    def _build_3d(self, nx: int, ny: int, nz: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
        """构建3D滤波矩阵"""
        ceil_rmin = int(ceil(rmin))
        nfilter = nx * ny * nz * ((2 * (ceil_rmin - 1) + 1) ** 3)
        iH = bm.zeros(nfilter, dtype=bm.int32)
        jH = bm.zeros(nfilter, dtype=bm.int32)
        sH = bm.zeros(nfilter, dtype=bm.float64)
        cc = 0
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    row = i * ny * nz + j * nz + k
                    ii1 = max(i - (ceil_rmin - 1), 0)
                    ii2 = min(i + ceil_rmin, nx)
                    jj1 = max(j - (ceil_rmin - 1), 0)
                    jj2 = min(j + ceil_rmin, ny)
                    kk1 = max(k - (ceil_rmin - 1), 0)
                    kk2 = min(k + ceil_rmin, nz)
                    
                    for ii in range(ii1, ii2):
                        for jj in range(jj1, jj2):
                            for kk in range(kk1, kk2):
                                col = ii * ny * nz + jj * nz + kk
                                fac = rmin - sqrt((i - ii)**2 + (j - jj)**2 + (k - kk)**2)
                                if fac > 0:
                                    iH[cc] = row
                                    jH[cc] = col
                                    sH[cc] = fac
                                    cc += 1
        
        return self._create_matrix(iH[:cc], jH[:cc], sH[:cc], nx * ny * nz)
    
    def _create_matrix(self, iH: TensorLike, jH: TensorLike, sH: TensorLike, size: int) -> Tuple[TensorLike, TensorLike]:
        """创建过滤矩阵和缩放向量"""
        H = COOTensor(
            indices=bm.astype(bm.stack((iH, jH), axis=0), bm.int32),
            values=sH,
            spshape=(size, size)
        )
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)
        return H, Hs