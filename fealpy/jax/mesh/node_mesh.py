
from typing import (Callable, Optional, Union,)

import jax
import jax.numpy as jnp
from jax import vmap
# from jax_md import space, partition

import numpy as np

from .mesh_base import MeshDS 
from .utils import Array, Dict
from jax import config

config.update("jax_enable_x64", True)

class NodeMesh(MeshDS):

    def __init__(self, node:Array, nodedata:Optional[Dict]=None, box=None):
        super().__init__(TD=0)
        self.node = node
 
        if nodedata is None:
            self.nodedata = {} # the data defined on node
        else:
            self.nodedata = nodedata


    def number_of_node(self):
        return self.ds.NN 
    
    def geo_dimension(self):
        return self.node.shape[1]

    def top_dimension(self):
        return self.top_dimension()
    '''
    def add_node_data(self, name:str, dtype=jnp.float64):
        NN = self.NN
        self.nodedata.update({names: jnp.zeros(NN, dtypes) 
                              for names, dtypes in zip(name, dtype)})
    '''
    
    def add_node_data(self, name: Union[str, list], data: Array):
        if isinstance(name, str):
            if name in self.nodedata:
                self.nodedata[name] = jnp.concatenate([self.nodedata[name], data], axis=0)
            else:
                self.nodedata[name] = data
        else:
            for n, d in zip(name, data):
                if n in self.nodedata:
                    self.nodedata[n] = jnp.concatenate([self.nodedata[n], d], axis=0)
                else:
                    self.nodedata[n] = d
    
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
    def from_tgv_domain(cls, box_size, dx=0.02, dy=0.02):
        rho0 = 1.0 #参考密度
        eta0 = 0.01 #参考动态粘度
        n = np.array((box_size / dx).round(), dtype=int)
        grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")

        r = jnp.array((jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx)
        NN = r.shape[0]
        tag = jnp.full(NN, 0, dtype=int)
        mv = jnp.zeros((NN, 2), dtype=jnp.float64)
        tv = jnp.zeros((NN, 2), dtype=jnp.float64)
        x = r[:, 0]
        y = r[:, 1]
        u0 = -jnp.cos(2.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y)
        v0 = jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)
        mv = mv.at[:,0].set(u0)
        mv = mv.at[:,1].set(v0)
        tv = mv
        volume = jnp.ones(NN, dtype=jnp.float64) * dx * dy
        rho = jnp.ones(NN, dtype=jnp.float64) * rho0
        mass = jnp.ones(NN, dtype=jnp.float64) * dx * dy * rho0
        eta = jnp.ones(NN, dtype=jnp.float64) * eta0

        nodedata = {
            "position": r,
            "tag": tag,
            "mv": mv,
            "tv": tv,
            "dmvdt": jnp.zeros_like(mv),
            "dtvdt": jnp.zeros_like(mv),
            "rho": rho,
            "mass": mass,
            "eta": eta,
        }
        return cls(r, nodedata=nodedata)

    @classmethod
    def from_heat_transfer_domain(cls, dx=0.02, dy=0.02):
        n_walls = 3 #墙壁层数
        rho0 = 1.0 #参考密度
        eta0 = 0.01 #参考动态粘度
        T0 = 1.0 #参考温度
        kappa0 = 7.313 #参考导热系数
        Cp0 = 305.27 #参考热容
        L,H = 1.0,0.2 
        hot_wall_half_width = 0.25 #温度一半长

        #墙粒子生成
        dxn1 = dx * n_walls
        n1 = np.array((np.array([L, dxn1]) / dx).round(), dtype=int)
        grid1 = np.meshgrid(range(n1[0]), range(n1[1]), indexing="xy")
        r1 = (jnp.vstack(list(map(jnp.ravel, grid1))).T + 0.5) * dx
        wall_b = r1.copy()
        wall_t = r1.copy() + np.array([0.0, H + dxn1])
        r_w = jnp.concatenate([wall_b, wall_t])
        
        #流体粒子生成
        n2 = np.array((np.array([L, H]) / dx).round(), dtype=int)
        grid2 = np.meshgrid(range(n2[0]), range(n2[1]), indexing="xy")
        r2 = (jnp.vstack(list(map(jnp.ravel, grid2))).T + 0.5) * dx
        r_f = np.array([0.0, 1.0]) * n_walls * dx + r2

        #设置标签
        '''
        0 fluid
        1 solid wall
        2 moving wall
        3 dirchilet wall
        '''
        tag_f = jnp.full(len(r_f), 0, dtype=int)
        tag_w = jnp.full(len(r_w), 1, dtype=int)
        r = np.array(np.concatenate([r_w, r_f]))
        tag = np.concatenate([tag_w, tag_f])

        #设置温度标签
        dx2n = dx * n_walls * 2
        _box_size = np.array([L, H + dx2n])
        mask_hot_wall = ((r[:, 1] < dx * n_walls) * (r[:, 0] < (_box_size[0] / 2) + \
                hot_wall_half_width) * (r[:, 0] > (_box_size[0] / 2) - hot_wall_half_width))
        tag = jnp.where(mask_hot_wall, 3, tag)
        
        NN_sum = r.shape[0]
        mv = jnp.zeros_like(r)
        rho = jnp.ones(NN_sum) * rho0
        mass = jnp.ones(NN_sum) * dx * dy * rho0
        eta = jnp.ones(NN_sum) * eta0
        temperature = jnp.ones(NN_sum) * T0
        kappa = jnp.ones(NN_sum) * kappa0
        Cp = jnp.ones(NN_sum) * Cp0

        nodedata = {
            "position": r,
            "tag": tag,
            "mv": mv,
            "tv": mv,
            "dmvdt": jnp.zeros_like(mv),
            "dtvdt": jnp.zeros_like(mv),
            "drhodt": jnp.zeros_like(rho),
            "rho": rho,
            "p": jnp.zeros_like(rho),
            "mass": mass,
            "eta": eta,
            "dTdt": jnp.zeros_like(rho),
            "T": temperature,
            "kappa": kappa,
            "Cp": Cp,
        }
        return cls(r, nodedata=nodedata)

    @classmethod
    def from_four_heat_transfer_domain(cls, dx=0.02, dy=0.02):
        n_walls = 3 #墙壁层数
        rho0 = 1.0 #参考密度
        eta0 = 0.01 #参考动态粘度
        T0 = 1.0 #参考温度
        kappa0 = 7.313 #参考导热系数
        Cp0 = 305.27 #参考热容
        L,H = 1.5,0.2
        velocity_wall = 0.3 #每段温度边界长度

        #墙粒子生成
        dxn1 = dx * n_walls
        n1 = np.array((np.array([L, dxn1]) / dx).round(), dtype=int)
        grid1 = np.meshgrid(range(n1[0]), range(n1[1]), indexing="xy")
        r1 = (jnp.vstack(list(map(jnp.ravel, grid1))).T + 0.5) * dx
        wall_b = r1.copy()
        wall_t = r1.copy() + np.array([0.0, H + dxn1])
        r_w = jnp.concatenate([wall_b, wall_t])

        #流体粒子生成
        n2 = np.array((np.array([L, H]) / dx).round(), dtype=int)
        grid2 = np.meshgrid(range(n2[0]), range(n2[1]), indexing="xy")
        r2 = (jnp.vstack(list(map(jnp.ravel, grid2))).T + 0.5) * dx
        r_f = np.array([0.0, 1.0]) * n_walls * dx + r2

        #设置标签
        '''
        0 fluid
        1 solid wall
        2 moving wall
        3 velocity wall
        '''
        r = np.array(np.concatenate([r_w, r_f])) 
        tag_f = jnp.full(len(r_f), 0, dtype=int)
        tag_w = jnp.full(len(r_w), 1, dtype=int)
        r = np.array(np.concatenate([r_w, r_f]))
        tag = np.concatenate([tag_w, tag_f])

        #设置温度粒子标签
        dx2n = dx * n_walls * 2
        _box_size = np.array([L, H + dx2n])
        mask_hot_wall = (
        ((r[:, 1] < dx * n_walls) | (r[:, 1] > H + dx * n_walls)) &
        (((r[:, 0] > 0.3) & (r[:, 0] < 0.6)) | ((r[:, 0] > 0.9) & (r[:, 0] < 1.2)))
    )
        tag = jnp.where(mask_hot_wall, 3, tag)

        NN_sum = r.shape[0]
        mv = jnp.zeros_like(r)
        rho = jnp.ones(NN_sum) * rho0
        mass = jnp.ones(NN_sum) * dx * dy * rho0
        eta = jnp.ones(NN_sum) * eta0
        temperature = jnp.ones(NN_sum) * T0
        kappa = jnp.ones(NN_sum) * kappa0
        Cp = jnp.ones(NN_sum) * Cp0

        nodedata = {
            "position": r,
            "tag": tag,
            "mv": mv,
            "tv": mv,
            "dmvdt": jnp.zeros_like(mv),
            "dtvdt": jnp.zeros_like(mv),
            "drhodt": jnp.zeros_like(rho),
            "rho": rho,
            "p": jnp.zeros_like(rho),
            "mass": mass,
            "eta": eta,
            "dTdt": jnp.zeros_like(rho),
            "T": temperature,
            "kappa": kappa,
            "Cp": Cp,
        }
        return cls(r, nodedata=nodedata)

    @classmethod
    def from_long_rectangular_cavity_domain(cls, init_domain, domain, uin, dx=1.25e-4):
        H = 1.5 * dx
        dy = dx
        rho0 = 737.54

        #fluid particles
        fp = jnp.mgrid[init_domain[0]:init_domain[1]:dx, \
            init_domain[2]+dx:init_domain[3]:dx].reshape(2,-1).T
        
        #wall particles
        x0 = jnp.arange(domain[0],domain[1],dx)

        bwp = jnp.column_stack((x0,np.full_like(x0,domain[2])))
        uwp = jnp.column_stack((x0,np.full_like(x0,domain[3])))
        wp = jnp.vstack((bwp,uwp))
        
        #dummy particles
        bdp = jnp.mgrid[domain[0]:domain[1]:dx, \
                domain[2]-dx:domain[2]-dx*4:-dx].reshape(2,-1).T
        udp = jnp.mgrid[domain[0]:domain[1]:dx, \
                domain[3]+dx:domain[3]+dx*3:dx].reshape(2,-1).T
        dp = jnp.vstack((bdp,udp))
        
        #gate particles
        gp = jnp.mgrid[-dx:-dx-4*H:-dx, \
                domain[2]+dx:domain[3]:dx].reshape(2,-1).T
        
        #tag
        '''
        fluid particles: 0 
        wall particles: 1 
        dummy particles: 2
        gate particles: 3
        '''
        tag_f = jnp.full((fp.shape[0],), 0, dtype=int)
        tag_w = jnp.full((wp.shape[0],), 1, dtype=int)
        tag_d = jnp.full((dp.shape[0],), 2,dtype=int)
        tag_g = jnp.full((gp.shape[0],), 3,dtype=int)

        r = jnp.vstack((fp, gp, wp, dp))
        NN = r.shape[0]
        tag = jnp.hstack((tag_f, tag_g, tag_w, tag_d))
        fg_v =  jnp.ones_like(jnp.vstack((fp, gp))) * uin
        wd_v =  jnp.zeros_like(jnp.vstack((wp, dp)))
        v = jnp.vstack((fg_v, wd_v))
        rho = jnp.ones(NN) * rho0
        mass = jnp.ones(NN) * dx * dy * rho0 
        nodedata = {
            "position": r,
            "tag": tag,
            "v": v,
            "rho": rho,
            "p": jnp.zeros_like(rho),
            "sound": jnp.zeros_like(rho),
            "mass": mass, 
        } 
        return cls(r, nodedata=nodedata)
        
    @classmethod
    def from_slip_stick_domain(cls, dx=0.02, dy=0.02):
        n_walls = 3 #墙壁层数
        rho0 = 1.0 #参考密度
        eta0 = 0.01 #参考动态粘度
        L,H = 1.5,0.2
        velocity_wall = 0.3 #每段速度边界长度

        #墙粒子生成
        dxn1 = dx * n_walls
        n1 = np.array((np.array([L, dxn1]) / dx).round(), dtype=int)
        grid1 = np.meshgrid(range(n1[0]), range(n1[1]), indexing="xy")
        r1 = (jnp.vstack(list(map(jnp.ravel, grid1))).T + 0.5) * dx
        wall_b = r1.copy()
        wall_t = r1.copy() + np.array([0.0, H + dxn1])
        r_w = jnp.concatenate([wall_b, wall_t])

        #流体粒子生成
        n2 = np.array((np.array([L, H]) / dx).round(), dtype=int)
        grid2 = np.meshgrid(range(n2[0]), range(n2[1]), indexing="xy")
        r2 = (jnp.vstack(list(map(jnp.ravel, grid2))).T + 0.5) * dx
        r_f = np.array([0.0, 1.0]) * n_walls * dx + r2

        #设置标签
        '''
        0 fluid
        1 solid wall
        2 moving wall
        3 velocity wall
        '''
        r = np.array(np.concatenate([r_w, r_f])) 
        tag_f = jnp.full(len(r_f), 0, dtype=int)
        tag_w = jnp.full(len(r_w), 1, dtype=int)
        r = np.array(np.concatenate([r_w, r_f]))
        tag = np.concatenate([tag_w, tag_f])

        #设置速度粒子标签
        dx2n = dx * n_walls * 2
        _box_size = np.array([L, H + dx2n])
        mask_hot_wall = (
        ((r[:, 1] < dx * n_walls) | (r[:, 1] > H + dx * n_walls)) &
        (((r[:, 0] > 0.3) & (r[:, 0] < 0.6)) | ((r[:, 0] > 0.9) & (r[:, 0] < 1.2)))
    )
        tag = jnp.where(mask_hot_wall, 3, tag)

        NN_sum = r.shape[0]
        mv = jnp.zeros_like(r)
        mv = jnp.where(tag[:, None] == 1, jnp.array([1.0, 0.0]), mv)
        rho = jnp.ones(NN_sum) * rho0
        mass = jnp.ones(NN_sum) * dx * dy * rho0
        eta = jnp.ones(NN_sum) * eta0

        nodedata = {
            "position": r,
            "tag": tag,
            "mv": mv,
            "tv": mv,
            "dmvdt": jnp.zeros_like(mv),
            "dtvdt": jnp.zeros_like(mv),
            "drhodt": jnp.zeros_like(rho),
            "rho": rho,
            "p": jnp.zeros_like(rho),
            "mass": mass,
            "eta": eta,
            "dTdt": jnp.zeros_like(rho),
        }
        return cls(r, nodedata=nodedata)

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
