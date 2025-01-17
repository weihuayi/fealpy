from typing import Union, Optional, Sequence, Tuple, Any, Dict

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger

from .mesh_base import MeshDS 

import jax.numpy as jnp
import jax
from jax_md import space, partition
from jax import vmap, lax

class NodeMesh(MeshDS):
    def __init__(self, node: TensorLike, nodedata: Optional[Dict] = None, itype: str = 'default_itype', ftype: str = 'default_ftype') -> None: 
        super().__init__(TD=0, itype=itype, ftype=ftype)
        '''
        note : Currently using jax's own jax_md, vmap, lax, vstack , ravel,
               full, where, mgrid, column_stack, full_like, hstack.
        '''
        self.node = node

        if nodedata is None:
            self.nodedata = {} 
        else:
            self.nodedata = nodedata

    def number_of_node(self) -> int:
        '''
        @brief Number of particles.
        '''
        return self.node.shape[0] 

    def geo_dimension(self) -> int:
        '''
        @brief Geometric dimension.
        '''
        return self.node.shape[1]

    def top_dimension(self) -> int:
        '''
        @brief Topological dimension.
        '''
        return 0

    def add_plot(self, axes, color='k', markersize=20):
        axes.set_aspect('equal')
        return axes.scatter(self.node[..., 0], self.node[..., 1], c=color, s=markersize)

    @classmethod
    def from_tgv_domain(cls, box_size, dx=0.02):
        rho0 = 1.0 #参考密度
        eta0 = 0.01 #参考动态粘度
        
        #n = bm.tensor((box_size / dx).round(), dtype=int)
        n = bm.astype(box_size // dx, bm.int32)
        grid = bm.meshgrid(bm.arange(n[0]), bm.arange(n[1]), indexing="xy")
        
        r = (bm.stack((grid[0].flatten(),grid[1].flatten()), 1) + 0.5) * dx
        
        NN = r.shape[0]
        tag = bm.zeros(NN)
        mv = bm.zeros((NN, 2), dtype=bm.float64)
        tv = bm.zeros((NN, 2), dtype=bm.float64)
        x = r[:, 0]
        y = r[:, 1]
        u0 = -bm.cos(2.0 * bm.pi * x) * bm.sin(2.0 * bm.pi * y)
        v0 = bm.sin(2.0 * bm.pi * x) * bm.cos(2.0 * bm.pi * y)
        if hasattr(mv, 'at'):  
            mv = mv.at[:, 0].set(u0)
            mv = mv.at[:, 1].set(v0)
        else:  
            mv[:, 0] = u0
            mv[:, 1] = v0
        tv = mv
        volume = bm.ones(NN, dtype=bm.float64) * dx **2
        rho = bm.ones(NN, dtype=bm.float64) * rho0
        mass = bm.ones(NN, dtype=bm.float64) * dx ** 2 * rho0
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

        #wall particles
        dxn1 = dx * n_walls
        n1 = bm.tensor((bm.tensor([L, dxn1]) / dx).round(), dtype=int)
        grid1 = bm.meshgrid(bm.arange(n1[0]), bm.arange(n1[1]), indexing="xy")
        r1 = (jnp.vstack(list(map(jnp.ravel, grid1))).T + 0.5) * dx
        wall_b = r1.copy()
        wall_t = r1.copy() + bm.tensor([0.0, H + dxn1])
        r_w = bm.concatenate([wall_b, wall_t])

        #fuild particles
        n2 = bm.tensor((bm.tensor([L, H]) / dx).round(), dtype=int)
        grid2 = bm.meshgrid(bm.arange(n2[0]), bm.arange(n2[1]), indexing="xy")
        r2 = (jnp.vstack(list(map(jnp.ravel, grid2))).T + 0.5) * dx
        r_f = bm.tensor([0.0, 1.0]) * n_walls * dx + r2

        #tag
        '''
        0 fluid
        1 solid wall
        2 moving wall
        3 dirchilet wall
        '''
        tag_f = jnp.full(len(r_f), 0, dtype=int)
        tag_w = jnp.full(len(r_w), 1, dtype=int)
        r = bm.tensor(bm.concatenate([r_w, r_f]))
        tag = bm.concatenate([tag_w, tag_f])

        dx2n = dx * n_walls * 2
        _box_size = bm.tensor([L, H + dx2n])
        mask_hot_wall = ((r[:, 1] < dx * n_walls) * (r[:, 0] < (_box_size[0] / 2) + \
                hot_wall_half_width) * (r[:, 0] > (_box_size[0] / 2) - hot_wall_half_width))
        tag = jnp.where(mask_hot_wall, 3, tag)
        
        NN_sum = r.shape[0]
        mv = bm.zeros_like(r)
        rho = bm.ones(NN_sum) * rho0
        mass = bm.ones(NN_sum) * dx * dy * rho0
        eta = bm.ones(NN_sum) * eta0
        temperature = bm.ones(NN_sum) * T0
        kappa = bm.ones(NN_sum) * kappa0
        Cp = bm.ones(NN_sum) * Cp0

        nodedata = {
            "position": r,
            "tag": tag,
            "mv": mv,
            "tv": mv,
            "dmvdt": bm.zeros_like(mv),
            "dtvdt": bm.zeros_like(mv),
            "drhodt": bm.zeros_like(rho),
            "rho": rho,
            "p": bm.zeros_like(rho),
            "mass": mass,
            "eta": eta,
            "dTdt": bm.zeros_like(rho),
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

        #wall particles
        dxn1 = dx * n_walls
        n1 = bm.tensor((bm.tensor([L, dxn1]) / dx).round(), dtype=int)
        grid1 = bm.meshgrid(bm.arange(n1[0]), bm.arange(n1[1]), indexing="xy")
        r1 = (jnp.vstack(list(map(jnp.ravel, grid1))).T + 0.5) * dx
        wall_b = r1.copy()
        wall_t = r1.copy() + bm.tensor([0.0, H + dxn1])
        r_w = bm.concatenate([wall_b, wall_t])

        #fuild particles
        n2 = bm.array((bm.array([L, H]) / dx).round(), dtype=int)
        grid2 = bm.meshgrid(bm.arange(n2[0]), bm.arange(n2[1]), indexing="xy")
        r2 = (jnp.vstack(list(map(jnp.ravel, grid2))).T + 0.5) * dx
        r_f = bm.tensor([0.0, 1.0]) * n_walls * dx + r2

        #tag
        '''
        0 fluid
        1 solid wall
        2 moving wall
        3 velocity wall
        '''
        r = bm.tensor(bm.concatenate([r_w, r_f])) 
        tag_f = jnp.full(len(r_f), 0, dtype=int)
        tag_w = jnp.full(len(r_w), 1, dtype=int)
        r = bm.tensor(bm.concatenate([r_w, r_f]))
        tag = bm.concatenate([tag_w, tag_f])

        dx2n = dx * n_walls * 2
        _box_size = bm.tensor([L, H + dx2n])
        mask_hot_wall = (
        ((r[:, 1] < dx * n_walls) | (r[:, 1] > H + dx * n_walls)) &
        (((r[:, 0] > 0.3) & (r[:, 0] < 0.6)) | ((r[:, 0] > 0.9) & (r[:, 0] < 1.2)))
    )
        tag = jnp.where(mask_hot_wall, 3, tag)

        NN_sum = r.shape[0]
        mv = bm.zeros_like(r)
        rho = bm.ones(NN_sum) * rho0
        mass = bm.ones(NN_sum) * dx * dy * rho0
        eta = bm.ones(NN_sum) * eta0
        temperature = bm.ones(NN_sum) * T0
        kappa = bm.ones(NN_sum) * kappa0
        Cp = bm.ones(NN_sum) * Cp0

        nodedata = {
            "position": r,
            "tag": tag,
            "mv": mv,
            "tv": mv,
            "dmvdt": bm.zeros_like(mv),
            "dtvdt": bm.zeros_like(mv),
            "drhodt": bm.zeros_like(rho),
            "rho": rho,
            "p": bm.zeros_like(rho),
            "mass": mass,
            "eta": eta,
            "dTdt": bm.zeros_like(rho),
            "T": temperature,
            "kappa": kappa,
            "Cp": Cp,
        }

        return cls(r, nodedata=nodedata)

    @classmethod
    def from_long_rectangular_cavity_domain(cls, init_domain=bm.tensor([0.0,0.005,0,0.005]), domain=bm.tensor([0,0.05,0,0.005]), uin=bm.tensor([5.0, 0.0]), dx=1.25e-4):
        H = 1.5 * dx
        dy = dx
        rho0 = 737.54

        #fluid particles
        fp = jnp.mgrid[init_domain[0]:init_domain[1]:dx, \
            init_domain[2]+dx:init_domain[3]:dx].reshape(2,-1).T

        #wall particles
        x0 = bm.arange(domain[0],domain[1],dx)

        bwp = jnp.column_stack((x0,jnp.full_like(x0,domain[2])))
        uwp = jnp.column_stack((x0,jnp.full_like(x0,domain[3])))
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
        fg_v =  bm.ones_like(jnp.vstack((fp, gp))) * uin
        wd_v =  bm.zeros_like(jnp.vstack((wp, dp)))
        v = jnp.vstack((fg_v, wd_v))
        rho = bm.ones(NN) * rho0
        mass = bm.ones(NN) * dx * dy * rho0

        nodedata = {
            "position": r,
            "tag": tag,
            "v": v,
            "rho": rho,
            "p": bm.zeros_like(rho),
            "sound": bm.zeros_like(rho),
            "mass": mass, 
        } 
        
        return cls(r, nodedata=nodedata)

    @classmethod
    def from_dam_break_domain(cls, dx=0.02, dy=0.02):
        pp = jnp.mgrid[dx:1+dx:dx, dy:2+dy:dy].reshape(2, -1).T

        #down
        bp0 = jnp.mgrid[0:4+dx:dx, 0:dy:dy].reshape(2, -1).T
        bp1 = jnp.mgrid[-dx/2:4+dx/2:dx, -dy/2:dy/2:dy].reshape(2, -1).T
        bp = jnp.vstack((bp0, bp1))

        #left
        lp0 = jnp.mgrid[0:dx:dx, dy:4+dy:dy].reshape(2, -1).T
        lp1 = jnp.mgrid[-dx/2:dx/2:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        lp = jnp.vstack((lp0, lp1))

        #right
        rp0 = jnp.mgrid[4:4+dx/2:dx, dy:4+dy:dy].reshape(2, -1).T
        rp1 = jnp.mgrid[4+dx/2:4+dx:dx, dy-dy/2:4+dy/2:dy].reshape(2, -1).T
        rp = jnp.vstack((rp0, rp1))

        boundaryp = jnp.vstack((bp, lp, rp))
        node = jnp.vstack((pp, boundaryp))

        return cls(node)
'''
class KDTree:
    def __init__(self, points):
        """
        初始化KDTree,输入为一个N x D的NumPy数组,表示N个D维的点集。
        """
        self.points = points
        self.tree = self._build_tree(points, depth=0)

    def _build_tree(self, points, depth):
        """
        递归构建KD-Tree。points是当前节点的点集,depth是当前树的深度。
        """
        if len(points) == 0:
            return None
        
        # 选择当前维度
        k = points.shape[1]  # 点的维度
        axis = depth % k  # 循环选择维度
        
        # 排序并选择中位点作为根节点
        points = points[points[:, axis].argsort()]
        median_idx = len(points) // 2
        median_point = points[median_idx]

        # 递归构建左子树和右子树
        left_tree = self._build_tree(points[:median_idx], depth + 1)
        right_tree = self._build_tree(points[median_idx + 1:], depth + 1)

        return {
            'point': median_point,
            'left': left_tree,
            'right': right_tree,
            'axis': axis
        }

    def range_query(self, query_points, radius, include_self=False):
        """
        查找每个查询点距离为radius以内的所有邻居点,并返回邻居粒子的索引和自身粒子的索引。
        query_points: (M, D) 形状的 NumPy 数组,表示M个查询点。
        radius: 查找邻居的半径
        include_self: 是否将查询点本身的索引加入到邻居结果中
        """
        neighbors_within_range = []
        self_indices = []  # 存储查询点的索引
        for i, query_point in enumerate(query_points):
            neighbors = self._range_query(self.tree, query_point, radius, depth=0, query_idx=i, include_self=include_self)
            neighbors_within_range.append(neighbors)
            self_indices.append([i] * len(neighbors))  # 每个查询点的索引重复以匹配邻居数量
        
        # 将查询点索引和邻居粒子的索引拆开，按索引顺序整理
        neighbors = [item for sublist in neighbors_within_range for item in sublist]
        indices = [item for sublist in self_indices for item in sublist]
        
        return bm.array(neighbors), bm.array(indices)

    def _range_query(self, node, query_point, radius, depth, query_idx, include_self):
        """
        递归查找范围查询，返回邻居粒子的索引。
        """
        if node is None:
            return []

        axis = node['axis']
        neighbors = []

        # 计算当前点和查询点的距离
        point = node['point']
        dist = bm.linalg.norm(point - query_point)

        # 如果当前点在查询范围内，加入结果
        if dist <= radius:
            # 如果include_self为True，允许加入查询点本身
            if include_self or not (point==query_point).all():  
                # 获取该点的索引
                neighbor_idx = bm.where(bm.all(self.points == point, axis=1))[0][0]
                neighbors.append(neighbor_idx)  # 添加邻居粒子的索引

        # 判断是否需要继续递归查找子树
        next_branch = None
        opposite_branch = None

        if query_point[axis] < point[axis]:
            next_branch = node['left']
            opposite_branch = node['right']
        else:
            next_branch = node['right']
            opposite_branch = node['left']

        # 递归查询邻居
        neighbors.extend(self._range_query(next_branch, query_point, radius, depth + 1, query_idx, include_self))

        # 如果对面子树有可能有点在范围内，继续查询
        if abs(query_point[axis] - point[axis]) < radius:
            neighbors.extend(self._range_query(opposite_branch, query_point, radius, depth + 1, query_idx, include_self))

        return neighbors
'''
class KDTree:
    def __init__(self, points):
        """
        初始化KDTree,输入为一个N x D的NumPy数组,表示N个D维的点集。
        """
        self.points = points
        self.tree = self._build_tree(points, depth=0)

    def _build_tree(self, points, depth):
        """
        递归构建KD-Tree。points是当前节点的点集,depth是当前树的深度。
        """
        if len(points) == 0:
            return None
        
        # 选择当前维度
        k = points.shape[1]  # 点的维度
        axis = depth % k  # 循环选择维度
        
        # 排序并选择中位点作为根节点
        points = points[points[:, axis].argsort()]
        median_idx = len(points) // 2
        median_point = points[median_idx]

        # 递归构建左子树和右子树
        left_tree = self._build_tree(points[:median_idx], depth + 1)
        right_tree = self._build_tree(points[median_idx + 1:], depth + 1)

        return {
            'point': median_point,
            'left': left_tree,
            'right': right_tree,
            'axis': axis
        }
        
    @staticmethod
    @bm.compile
    def range_query(self, query_points, radius, include_self=False):
        """
        查找每个查询点距离为radius以内的所有邻居点,并返回邻居粒子的索引和自身粒子的索引。
        query_points: (M, D) 形状的 NumPy 数组,表示M个查询点。
        radius: 查找邻居的半径
        include_self: 是否将查询点本身的索引加入到邻居结果中
        """
        neighbors_within_range = []
        self_indices = []  # 存储查询点的索引
        for i, query_point in enumerate(query_points):
            neighbors = self._range_query(self.tree, query_point, radius, depth=0, query_idx=i, include_self=include_self)
            neighbors_within_range.append(neighbors)
            self_indices.append([i] * len(neighbors))  # 每个查询点的索引重复以匹配邻居数量
        
        # 将查询点索引和邻居粒子的索引拆开，按索引顺序整理
        neighbors = [item for sublist in neighbors_within_range for item in sublist]
        indices = [item for sublist in self_indices for item in sublist]
        
        return bm.array(neighbors), bm.array(indices)

    @staticmethod
    @bm.compile
    def _range_query(self, node, query_point, radius, depth, query_idx, include_self):
        if node is None:
            return []

        axis = node['axis']
        neighbors = []
        point = node['point']
        dist = bm.linalg.norm(point - query_point)

        # 剪枝策略：距离超过半径时，停止搜索
        if dist <= radius:
            if include_self or not (point==query_point).all():
                neighbor_idx = bm.where(bm.all(self.points == point, axis=1))[0][0]
                neighbors.append(neighbor_idx)

        next_branch = None
        opposite_branch = None
        if query_point[axis] < point[axis]:
            next_branch = node['left']
            opposite_branch = node['right']
        else:
            next_branch = node['right']
            opposite_branch = node['left']

        # 递归查询
        neighbors.extend(self._range_query(next_branch, query_point, radius, depth + 1, query_idx, include_self))

        # 剪枝：如果查询点和当前节点的差异已经大于半径，停止递归查询对面子树
        if abs(query_point[axis] - point[axis]) < radius:
            neighbors.extend(self._range_query(opposite_branch, query_point, radius, depth + 1, query_idx, include_self))

        return neighbors
