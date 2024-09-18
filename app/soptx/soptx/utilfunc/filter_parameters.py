from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.typing import TensorLike, Tuple
from fealpy.experimental.sparse import COOTensor

from builtins import int, float
from math import ceil, sqrt

def compute_filter_2d(nx: int, ny: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
    """
    Compute the filter matrix for the topology optimization.

    This function calculates the filter matrix and its scaling vector
    based on the specified filter radius. The filter is used to control
    the minimum length scale of the optimized structure, ensuring a
    more manufacturable design.

    Args:
        nx (int): The number of elements in the x-direction of the mesh.
        ny (int): The number of elements in the y-direction of the mesh.
        rmin (int): The filter radius, which controls the minimum feature size.

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

def compute_filter_3d(nx: int, ny: int, nz: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
    """
    Compute the filter matrix for 3D topology optimization.

    This function calculates the filter matrix and its scaling vector
    based on the specified filter radius for a 3D problem. The filter is used to control
    the minimum length scale of the optimized structure, ensuring a
    more manufacturable design.

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

def apply_filter(ft: int, rho: TensorLike, dce: TensorLike, dve: TensorLike, 
                H: TensorLike, Hs: TensorLike) -> Tuple[TensorLike, TensorLike]:
    """
    Apply the filter to the sensitivities.

    Args:
        ft (int): Filter type, 0 for sensitivity filter, 1 for density filter.
        rho (TensorLike): The density distribution of the material.
        dc (TensorLike): The sensitivity of the objective function.
        dv (TensorLike): The sensitivity of the volume constraint.
        H (TensorLike): The filter matrix.
        Hs (TensorLike): The scaling vector for the filter.

    Returns:
        tuple: Filtered sensitivity of the objective function and volume constraint.
    """

    if ft == 0:
        rho_dce = bm.multiply(rho, dce)
        filtered_dce = H.matmul(rho_dce)
        dce[:] = filtered_dce / Hs / bm.maximum(bm.array(0.001), rho)
        # dce[:] = filtered_dce / Hs / bm.set_at(rho[:], rho[:] < 0.001, 0.001)
    elif ft == 1:
        dce[:] = H.to_dense() * (dce / Hs)
        dve[:] = H.to_dense() * (dve / Hs)
    
    return dce, dve
