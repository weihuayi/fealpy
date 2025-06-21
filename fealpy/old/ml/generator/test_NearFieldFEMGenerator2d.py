import numpy as np
from numpy.typing import NDArray
from math import sqrt, pi
from fealpy.ml.generator import NearFieldDataFEMGenerator2d
from fealpy.ml.sampler import CircleCollocator


domain = [-6, 6, -6, 6]
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'
d = [[-sqrt(0.5), sqrt(0.5)]]
k = [2*pi]

def levelset(p: NDArray, centers: NDArray, radius: NDArray):
    """
    Calculate level set function value.
    """
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
    ret = np.min(dis - radius[None, :], axis=-1) # (N, )
    return ret.reshape(struct)

reciever_points = CircleCollocator(0, 0, 5).run(50)
reciever_points = reciever_points.detach().numpy()
cirs = np.array([[0.4, -0.6, 0.2],
                 [0.6, -0.5, 0.1],
                 [0.3, 0.2, 0.3]], dtype=np.float64)

centers = cirs[:, 0:2]
radius = cirs[:, 2]
ls_fn = lambda p: levelset(p, centers, radius)

generator = NearFieldDataFEMGenerator2d(domain=domain,
                                        mesh='UniformMesh',
                                        nx=100,
                                        ny=100,
                                        p=1,
                                        q=3,
                                        u_inc=u_inc,
                                        levelset=ls_fn,
                                        d=d,
                                        k=k,
                                        reciever_points=reciever_points)
generator.visualization_of_nearfield_data(k=2*pi, d=[-sqrt(0.5), sqrt(0.5)])
