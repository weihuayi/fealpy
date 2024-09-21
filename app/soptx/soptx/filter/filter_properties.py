from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike, Tuple
from fealpy.experimental.sparse import COOTensor
from fealpy.experimental.mesh.uniform_mesh_2d import UniformMesh2d
from fealpy.experimental.mesh.uniform_mesh_3d import UniformMesh3d

from math import ceil, sqrt

class FilterProperties:
    def __init__(self, mesh, rmin: float, ft: int):
        """
        Initialize the filter properties based on the mesh.

        Args:
            mesh: The mesh object which contains information about the grid.
            rmin (float): The filter radius, which controls the minimum feature size.
            ft (int): The filter type, 0 for sensitivity filter, 1 for density filter.
        """
        self.ft = ft

        if not isinstance(mesh, (UniformMesh2d, UniformMesh3d)):
            raise TypeError("mesh must be an instance of UniformMesh2d or UniformMesh3d.")

        nx, ny = mesh.nx, mesh.ny
        nz = getattr(mesh, 'nz', None)

        if nz is not None:
            self.H, self.Hs = self._compute_filter_3d(nx, ny, nz, rmin)
        else:
            self.H, self.Hs = self._compute_filter_2d(nx, ny, rmin)

    def _compute_filter_2d(self, nx: int, ny: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
        """
        Compute the filter matrix for 2D topology optimization.

        Args:
            nx (int): The number of elements in the x-direction of the mesh.
            ny (int): The number of elements in the y-direction of the mesh.
            rmin (float): The filter radius, which controls the minimum feature size.

        Returns:
            Tuple[COOTensor, bm.ndarray]: A tuple containing the filter matrix (H)
                                          in COO format and the scaling vector (Hs).
        """
        if nx <= 0 or ny <= 0:
            raise ValueError("The number of elements in both x and y directions (nx, ny) must be positive integers.")
        if rmin <= 0:
            raise ValueError("The filter radius (rmin) must be a positive integer.")

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
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = max(0.0, fac)
                        cc += 1
        H = COOTensor(indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32), 
                    values=sH[:cc], 
                    spshape=(nx * ny, nx * ny))
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)

        return H, Hs

    def _compute_filter_3d(self, nx: int, ny: int, nz: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
        """
        Compute the filter matrix for 3D topology optimization.

        Args:
            nx (int): The number of elements in the x-direction of the mesh.
            ny (int): The number of elements in the y-direction of the mesh.
            nz (int): The number of elements in the z-direction of the mesh.
            rmin (float): The filter radius, which controls the minimum feature size.

        Returns:
            Tuple[COOTensor, bm.ndarray]: A tuple containing the filter matrix (H)
                                          in COO format and the scaling vector (Hs).
        """
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError("The number of elements in all directions (nx, ny, nz) must be positive integers.")
        if rmin <= 0:
            raise ValueError("The filter radius (rmin) must be a positive number.")

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
                                iH[cc] = row
                                jH[cc] = col
                                sH[cc] = max(0.0, fac)
                                cc += 1

        H = COOTensor(indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32), 
                      values=sH[:cc], 
                      spshape=(nx * ny * nz, nx * ny * nz))
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)

        return H, Hs
