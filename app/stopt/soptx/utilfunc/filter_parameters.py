from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.typing import TensorLike, Tuple
from fealpy.experimental.sparse import COOTensor

from builtins import int, float
from math import ceil, sqrt

def compute_filter(nx: int, ny: int, rmin: float) -> Tuple[TensorLike, TensorLike]:
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
                                      in compressed sparse row (CSR) format and
                                      the scaling vector (Hs).
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("The number of elements in both x and y directions (nx, ny) must be positive integers.")
    if rmin <= 0:
        raise ValueError("The filter radius (rmin) must be a positive integer.")
    
    nfilter = int(nx * ny * ((2 * (ceil(rmin) - 1) + 1) ** 2))
    iH = bm.zeros(nfilter, dtype=bm.int32)
    jH = bm.zeros(nfilter, dtype=bm.int32)
    sH = bm.zeros(nfilter, dtype=bm.float64)
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

    H = COOTensor(indices=bm.astype(bm.stack((iH, jH), axis=0), bm.int32), values=sH, 
                                    spshape=(nx * ny, nx * ny)).tocsr()
    Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)

    return H, Hs