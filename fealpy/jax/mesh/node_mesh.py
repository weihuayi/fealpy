
from typing import (Callable, Optional, Union,)

import jax
import jax.numpy as jnp
from ..sph.jax_md import space, partition
from ..sph.jax_md.partition import (NeighborListFormat)
from jax import vmap

from .mesh_base import MeshDS 
from .utils import Array, Dict
from jax.config import config

config.update("jax_enable_x64", True)

class NodeMesh(MeshDS):

    def __init__(self, node:Array, nodedata:Optional[Dict]=None, box=None):
        super().__init__(TD=0)
        self.node = node
 
        if nodedata is None:
            self.nodedata = {} # the data defined on node
        else:
            self.nodedata = nodedata

    def geo_dimension(self):
        return self.node.shape[1]

    def top_dimension(self):
        return self.top_dimension()
    
    def add_node_data(self, name:str, dtype=jnp.float64):
        NN = self.NN
        self.nodedata.update({names: jnp.zeros(NN, dtypes) 
                              for names, dtypes in zip(name, dtype)})

    def set_node_data(self, name, val):
        self.nodedata[name] = self.nodedata[name].at[:].set(val)

    def add_plot(self, axes, color='k', markersize=20):
        axes.set_aspect('equal')
        return axes.scatter(self.node[..., 0], self.node[..., 1], c=color, s=markersize)

    def neighbors(self, box_size, h):
        """
        参数:
        - box_size: 模拟盒子的大小
        - h: 平滑长度
        返回:
        - index: 每个粒子的邻近粒子索引
        - indptr: 每行邻近索引的计数
        """
        # 定义邻近列表的参数
        displacement, shift = space.periodic(box_size)
        neighbor_fn = partition.neighbor_list(displacement, box_size, h)
        # 初始化邻近列表，用于分配内存
        nbrs = neighbor_fn.allocate(self.node)
        # 更新邻近列表
        nbrs = neighbor_fn.update(self.node, nbrs)
        neighbor = nbrs.idx
        num = self.node.shape[0]
        index = jax.vmap(lambda idx, row: jnp.hstack([row, jnp.array([idx])]))(jnp.arange(neighbor.shape[0]), neighbor)
        row_len = jnp.sum(index != num,axis=1)
        indptr = jax.lax.scan(lambda carry, x: (carry + x, carry + x), 0, row_len)[1]
        indptr = jnp.concatenate((jnp.array([0]), indptr))
        index = index[index != num]
        return index, indptr
    
    def interpolate(self, u, kernel, neighbor, h):
        """
        参数:
        - u: 函数值
        - kernel: 核函数
        - neighbor: 邻近索引
        - h: 光滑长度
        返回:
        - kernel_function: 核函数插值后的值
        - kernel_grad : 核函数梯度插值后的值
        """
        kernel_function = []
        kernel_grad = []
        for i, value in neighbor.items():
            rho = len(value['indices'])
            m = jnp.pi * h**2
            if rho == 0:
                kernel_function.append(0)
                kernel_grad.append(0)
            else:
                idx = [item.item() for item in value['indices']]
                dist = jnp.array([item.item() for item in value['distances']])
                w = kernel.value(dist)
                k = jnp.dot((m/rho)*u[jnp.array(idx)],w)
                kernel_function.append(k)
          
                reduce = self.node[1,:] - self.node[idx,:]
                dw = vmap(kernel.grad_value)(dist)[:,jnp.newaxis]*reduce
                dk = jnp.dot((m/rho)*u[jnp.array(idx)],dw)
                kernel_grad.append(dk)
        return kernel_function, kernel_grad
    
    @classmethod
    def from_tgv_domain(cls, box_size, dx=0.02):
        n = np.array((box_size / dx).round(), dtype=int)
        grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
        r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
        return cls(r)

    @classmethod
    def from_ringshaped_channel_domain(cls, dx=0.02, dy=0.02):
        #下
        dp0 = jnp.mgrid[0:1.38+dx:dx, -dy:dy:dy].reshape(2, -1).T
        dp3 = jnp.mgrid[2.82:4.2+dx:dx, -dy:dy:dy].reshape(2, -1).T
        
        start_diff0 = jnp.array([1,0])-jnp.array([2.1,1.1])
        start_angle0 = jnp.arctan2(start_diff0[1], start_diff0[0])
        end_diff0 = jnp.array([3.2,0])-jnp.array([2.1,1.1])
        end_angle0 = jnp.arctan2(end_diff0[1], end_diff0[0])
        if end_angle0 < start_angle0:
            end_angle0 += 2 * jnp.pi
        angles0 = jnp.arange(start_angle0, end_angle0+dx, dx)
        x0 = jnp.array([2.1,1.1])[0] + jnp.cos(angles0)
        y0 = jnp.array([2.1,1.1])[1] + jnp.sin(angles0)
        dp1 = jnp.stack([x0, y0], axis=-1)-jnp.array([0,0.4])
        dp2 = jnp.stack([x0, y0], axis=-1)-jnp.array([0,0.4+dx])
        
        dp = jnp.vstack((dp0,dp1,dp2,dp3))
        #上
        up0 = jnp.mgrid[0:1.38+dx:dx, 1:1+dy:dy].reshape(2, -1).T
        up3 = jnp.mgrid[2.82:4.2+dx:dx, 1:1+dy:dy].reshape(2, -1).T
        
        start_diff1 = jnp.array([1,0])-jnp.array([2.1,1.1])
        start_angle1 = jnp.arctan2(start_diff1[1], start_diff1[0])
        end_diff1 = jnp.array([3.2,0])-jnp.array([2.1,1.1])
        end_angle1 = jnp.arctan2(end_diff1[1], end_diff1[0])
        if end_angle1 < start_angle1:
            end_angle1 += 2 * jnp.pi
        angles1 = jnp.arange(start_angle1+jnp.pi, end_angle1+jnp.pi+dx, dx)
        x1 = jnp.array([2.1,1.1])[0] + jnp.cos(angles1)
        y1 = jnp.array([2.1,1.1])[1] + jnp.sin(angles1)
        up1 = jnp.stack([x1, y1], axis=-1)-jnp.array([0,0.8])
        up2 = jnp.stack([x1, y1], axis=-1)-jnp.array([0,0.8-dx])
        
        up = jnp.vstack((up0,up1,up2,up3))
        #芯型
        circumference = 2 * jnp.pi * 0.25
        num = int(circumference / dx)
        angles = jnp.linspace(0, 2 * jnp.pi, num, endpoint=False)
        x = jnp.array([2.1,0.55])[0] + 0.25 * jnp.cos(angles)
        y = jnp.array([2.1,0.55])[1] + 0.25 * jnp.sin(angles)
        round = jnp.stack([x, y], axis=-1)
        return cls(jnp.vstack((dp, up,round)))

    @classmethod
    def from_dam_break_domain(cls, dx=0.02, dy=0.02):
        pp = jnp.mgrid[dx:1+dx:dx, dy:2+dy:dy].reshape(2, -1).T        
        # 下
        bp0 = jnp.mgrid[0:4+dx:dx, 0:dy:dy].reshape(2, -1).T
        bp1 = jnp.mgrid[-dx/2:4+dx/2:dx, -dy/2:dy/2:dy].reshape(2, -1).T
        bp = jnp.vstack((bp0, bp1))
        # 左
        lp0 = jnp.mgrid[0:dx:dx, dy:4+dy:dy].reshape(2, -1).T
        lp1 = jnp.mgrid[-dx/2:dx/2:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        lp = jnp.vstack((lp0, lp1))
        # 右
        rp0 = jnp.mgrid[4:4+dx/2:dx, dy:4+dy:dy].reshape(2, -1).T
        rp1 = jnp.mgrid[4+dx/2:4+dx:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        rp = jnp.vstack((rp0, rp1))
        
        boundaryp = jnp.vstack((bp, lp, rp))
        node = jnp.vstack((pp, boundaryp))
        result = cls(node)
        result.is_boundary = result.is_boundary.at[pp.shape[0]:].set(True)
        return result
