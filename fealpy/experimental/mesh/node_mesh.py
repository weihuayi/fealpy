from typing import Union, Optional, Sequence, Tuple, Any, Dict

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import MeshDS 
import jax.numpy as jnp
class NodeMesh(MeshDS):
    def __init__(self, node: TensorLike, nodedata:Optional[Dict]=None) -> None: 
        super().__init__(TD=0)
        self.node = node

        if nodedata is None:
            self.nodedata = {} 
        else:
            self.nodedata = nodedata

    def number_of_node(self) -> int:
        return self.node.shape[0] 

    def geo_dimension(self) -> int:
        return self.node.shape[1]

    def top_dimension(self) -> int:
        return 0

    def add_node_data(self, name:Union[str, list], data: TensorLike) -> Union:
        if isinstance(name, str):
            if name in self.nodedata:
                self.nodedata[name] = bm.concatenate([self.nodedata[name], bm.tensor(data)], axis=0)
            else:
                self.nodedata[name] = bm.tensor(data)
        else:
            for n, d in zip(name, data):
                if n in self.nodedata:
                    self.nodedata[n] = bm.concatenate([self.nodedata[n], bm.tensor(d)], axis=0)
                else:
                    self.nodedata[n] = bm.tensor(d)
    
    def set_node_data(self, name:Union[str, list], data:TensorLike) -> Union:
        self.nodedata[name] = self.nodedata[name].at[:].set(data)

    def add_plot(self, axes, color='k', markersize=20):
        axes.set_aspect('equal')
        return axes.scatter(self.node[..., 0], self.node[..., 1], c=color, s=markersize)

    @classmethod
    def from_tgv_domain(cls, box_size, dx=0.02, dy=0.02):
        rho0 = 1.0 #参考密度
        eta0 = 0.01 #参考动态粘度
        n = bm.tensor((box_size / dx).round(), dtype=int)
        grid = bm.meshgrid(bm.arange(n[0]), bm.arange(n[1]), indexing="xy")
        
        r = bm.tensor((jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx)
        NN = r.shape[0]
        tag = jnp.full(NN, 0, dtype=int)
        mv = bm.zeros((NN, 2), dtype=bm.float64)
        tv = bm.zeros((NN, 2), dtype=bm.float64)
        x = r[:, 0]
        y = r[:, 1]
        u0 = -bm.cos(2.0 * bm.pi * x) * bm.sin(2.0 * bm.pi * y)
        v0 = bm.sin(2.0 * bm.pi * x) * bm.cos(2.0 * bm.pi * y)
        mv = mv.at[:,0].set(u0)
        mv = mv.at[:,1].set(v0)
        tv = mv
        volume = bm.ones(NN, dtype=bm.float64) * dx * dy
        rho = bm.ones(NN, dtype=bm.float64) * rho0
        mass = bm.ones(NN, dtype=bm.float64) * dx * dy * rho0
        eta = bm.ones(NN, dtype=bm.float64) * eta0

        nodedata = {
            "position": r,
            "tag": tag,
            "mv": mv,
            "tv": tv,
            "dmvdt": bm.zeros_like(mv),
            "dtvdt": bm.zeros_like(mv),
            "rho": rho,
            "mass": mass,
            "eta": eta,
        }
        
        return cls(r, nodedata=nodedata)
