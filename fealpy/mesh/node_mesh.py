from typing import Union, Optional, Sequence, Tuple, Any, Dict

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger
from .mesh_base import MeshDS 
from fealpy.backend import TensorLike
from scipy.spatial import cKDTree

# Types
Box = TensorLike

class NodeMesh(MeshDS):
    def __init__(self, node: TensorLike, nodedata: Optional[Dict] = None, itype: str = 'default_itype', ftype: str = 'default_ftype') -> None: 
        super().__init__(TD=0, itype=itype, ftype=ftype)
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

        n = bm.astype(box_size // dx, bm.int32)
        grid = bm.meshgrid(bm.arange(n[0],dtype=bm.float64), bm.arange(n[1], dtype=bm.float64), indexing="xy")
        
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
            "dx": dx,
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
        dxn1 = bm.array(dx * n_walls, dtype=bm.float64)
        n1 = bm.astype(bm.array([L, dxn1]) / dx, bm.int32)
        grid1 = bm.meshgrid(bm.arange(n1[0]), bm.arange(n1[1]), indexing="xy")
        r1 = (bm.stack((grid1[0].flatten(),grid1[1].flatten()), 1) + 0.5) * dx
        wall_b = bm.copy(r1)
        wall_t = bm.copy(r1) + bm.array([0.0, H + dxn1], dtype=bm.float64)
        r_w = bm.concatenate([wall_b, wall_t])

        #fuild particles
        n2 = bm.astype(bm.array([L, H]) / dx, bm.int32)
        grid2 = bm.meshgrid(bm.arange(n2[0]), bm.arange(n2[1]), indexing="xy")
        r2 = (bm.stack((grid2[0].flatten(),grid2[1].flatten()), 1) + 0.5) * dx
        r_f = bm.astype(bm.array([0.0, 1.0]) * n_walls * dx + r2, bm.float64)

        #tag:0-fluid,1-solid wall,2-moving wall,3-dirchilet wall
        tag_f = bm.full((r_f.shape[0],), 0, dtype=bm.int32)
        tag_w = bm.full((r_w.shape[0],), 1, dtype=bm.int32)
        
        r = bm.concatenate([r_w, r_f])
        tag = bm.concatenate([tag_w, tag_f])

        dx2n = bm.array(dx * n_walls * 2, dtype=bm.float64)
        _box_size = bm.array([L, H + dx2n], dtype=bm.float64)
        mask_hot_wall = ((r[:, 1] < dx * n_walls) * (r[:, 0] < (_box_size[0] / 2) + \
                hot_wall_half_width) * (r[:, 0] > (_box_size[0] / 2) - hot_wall_half_width))
        tag = bm.where(mask_hot_wall, 3, tag)
        
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
            "dx": dx,
            
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
        dxn1 = bm.array(dx * n_walls, dtype=bm.float64)
        n1 = bm.astype(bm.array([L, dxn1]) / dx, bm.int32)
        grid1 = bm.meshgrid(bm.arange(n1[0]), bm.arange(n1[1]), indexing="xy")
        r1 = (bm.stack((grid1[0].flatten(),grid1[1].flatten()), 1) + 0.5) * dx
        wall_b = bm.copy(r1)
        wall_t = bm.copy(r1) + bm.array([0.0, H + dxn1], dtype=bm.float64)
        r_w = bm.concatenate([wall_b, wall_t])

        #fuild particles
        n2 = bm.astype(bm.array([L, H]) / dx, bm.int32)
        grid2 = bm.meshgrid(bm.arange(n2[0]), bm.arange(n2[1]), indexing="xy")
        r2 = (bm.stack((grid2[0].flatten(),grid2[1].flatten()), 1) + 0.5) * dx
        r_f = bm.astype(bm.array([0.0, 1.0]) * n_walls * dx + r2, bm.float64)

        #tag:0-fluid,1-solid wall,2-moving wall,3-dirchilet wall
        tag_f = bm.full((r_f.shape[0],), 0, dtype=bm.int32)
        tag_w = bm.full((r_w.shape[0],), 1, dtype=bm.int32)

        r = bm.concatenate([r_w, r_f])
        tag = bm.concatenate([tag_w, tag_f])

        dx2n = bm.array(dx * n_walls * 2, dtype=bm.float64)
        _box_size = bm.array([L, H + dx2n], dtype=bm.float64)
        mask_hot_wall = (
        ((r[:, 1] < dx * n_walls) | (r[:, 1] > H + dx * n_walls)) &
        (((r[:, 0] > 0.3) & (r[:, 0] < 0.6)) | ((r[:, 0] > 0.9) & (r[:, 0] < 1.2)))
    )
        tag = bm.where(mask_hot_wall, 3, tag)

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
            "dx": dx,
        }

        return cls(r, nodedata=nodedata)

    @classmethod
    def from_pipe_domain(cls, domain, init_domain, H, dx=1.25e-4):
        u_in = bm.array([5.0, 0.0], dtype=bm.float64) 
        rho_0 = 737.54

        f_x = bm.arange(init_domain[0], init_domain[1], dx, dtype=bm.float64)
        f_y = bm.arange(init_domain[2] + dx, init_domain[3], dx, dtype=bm.float64)
        fy, fx = bm.meshgrid(f_y, f_x, indexing='xy')
        fp = bm.stack((fx.ravel(), fy.ravel()), axis=1)
        f_tag = bm.zeros(len(fp), dtype=bm.int64)
        
        x0 = bm.arange(domain[0], domain[1], dx, dtype=bm.float64)
        bwp = bm.stack((x0, bm.full_like(x0, domain[2])), axis=1)
        uwp = bm.stack((x0, bm.full_like(x0, domain[3])), axis=1)
        wp = bm.concatenate((bwp, uwp), axis=0)
        w_tag = bm.ones(len(wp), dtype=bm.int64)
        
        d_xb = bm.arange(domain[0], domain[1], dx, dtype=bm.float64)
        d_yb = bm.arange(domain[2] - dx, domain[2] - dx * 4, -dx, dtype=bm.float64)
        dyb, dxb = bm.meshgrid(d_yb, d_xb, indexing='xy')
        bdp = bm.stack((dxb.ravel(), dyb.ravel()), axis=1)
        d_yu = bm.arange(domain[3] + dx, domain[3] + dx * 3, dx, dtype=bm.float64)
        dyu, dxu = bm.meshgrid(d_yu, d_xb, indexing='xy')
        udp = bm.stack((dxu.ravel(), dyu.ravel()), axis=1)
        dp = bm.concatenate((bdp, udp), axis=0)
        d_tag = bm.full((len(dp),), 2, dtype=bm.int64)

        g_x = bm.arange(-dx, -dx - 4 * H, -dx, dtype=bm.float64)
        g_y = bm.arange(domain[2] + dx, domain[3], dx, dtype=bm.float64)
        gy, gx = bm.meshgrid(g_y, g_x, indexing='xy')
        gp = bm.stack((gx.ravel(), gy.ravel()), axis=1)
        g_tag = bm.full((len(gp),), 3, dtype=bm.int64)

        r = bm.concatenate((fp, wp, dp, gp), axis=0)
        tag = bm.concatenate((f_tag, w_tag, d_tag, g_tag), axis=0)
        u = bm.zeros((len(r), 2), dtype=bm.float64)
        u = bm.set_at(u, (tag == 0)|(tag == 3), u_in)
        rho = bm.full((len(r),), rho_0, dtype=bm.float64)
        m0 = rho_0 * (init_domain[1] - init_domain[0]) * (init_domain[3] - init_domain[2]) / fp.shape[0]
        m1 = rho_0 * 6 * dx * (domain[3] - domain[2]) / gp.shape[0]
        mass0 = bm.full((len(r[tag != 3]),), m0, dtype=bm.float64)
        mass1 = bm.full((len(r[tag == 3]),), m1, dtype=bm.float64)
        mass = bm.concatenate((mass0, mass1), axis=0)
        
        nodedata = {
            "position": r,
            "tag": tag,
            "u": u,
            "dudt": bm.zeros_like(u),
            "rho": rho,
            "drhodt": bm.zeros_like(rho),
            "p": bm.zeros_like(rho),
            "sound": bm.zeros_like(rho),
            "mass": mass,
            "mu": bm.zeros_like(rho),
            "drdt": bm.zeros_like(r),
            "dx": dx,
        }
        return cls(r, nodedata=nodedata)

    @classmethod
    def from_pipe_domain0(cls, domain, init_domain, H, dx=1.25e-4):
        u_in = bm.array([5.0, 0.0], dtype=bm.float64) 
        rho_0 = 737.54

        f_x = bm.arange(init_domain[0], init_domain[1], dx, dtype=bm.float64)
        f_y = bm.arange(init_domain[2] + dx, init_domain[3], dx, dtype=bm.float64)
        fy, fx = bm.meshgrid(f_y, f_x, indexing='xy')
        fp = bm.stack((fx.ravel(), fy.ravel()), axis=1)
        f_tag = bm.zeros(len(fp), dtype=bm.int64)
        
        x0 = bm.arange(domain[0], domain[1]/2, dx, dtype=bm.float64)
        bwp = bm.stack((x0, bm.full_like(x0, domain[2])), axis=1)
        uwp = bm.stack((x0, bm.full_like(x0, domain[3])), axis=1)
        y0 = bm.arange(domain[2] - dx * 3, domain[3] + dx * 4, dx, dtype=bm.float64)
        rwp = bm.concatenate((bm.full((len(y0), 1), domain[1]/2), y0.reshape(-1, 1)), axis=1)
        wp = bm.concatenate((bwp, uwp, rwp), axis=0)
        w_tag = bm.ones(len(wp), dtype=bm.int64)
        
        d_xb = bm.arange(domain[0], domain[1]/2, dx, dtype=bm.float64)
        d_yb = bm.arange(domain[2] - dx, domain[2] - dx * 4, -dx, dtype=bm.float64)
        dyb, dxb = bm.meshgrid(d_yb, d_xb, indexing='xy')
        bdp = bm.stack((dxb.ravel(), dyb.ravel()), axis=1)
        d_yu = bm.arange(domain[3] + dx, domain[3] + dx * 3, dx, dtype=bm.float64)
        dyu, dxu = bm.meshgrid(d_yu, d_xb, indexing='xy')
        udp = bm.stack((dxu.ravel(), dyu.ravel()), axis=1)

        x1 = bm.arange(domain[1]/2+dx, domain[1]/2 + 3*dx, dx, dtype=bm.float64)
        dyr, dxr = bm.meshgrid(y0, x1, indexing='xy')
        bdr = bm.stack((dxr.ravel(), dyr.ravel()), axis=1)

        dp = bm.concatenate((bdp, udp, bdr), axis=0)
        d_tag = bm.full((len(dp),), 2, dtype=bm.int64)

        g_x = bm.arange(-dx, -dx - 4 * H, -dx, dtype=bm.float64)
        g_y = bm.arange(domain[2] + dx, domain[3], dx, dtype=bm.float64)
        gy, gx = bm.meshgrid(g_y, g_x, indexing='xy')
        gp = bm.stack((gx.ravel(), gy.ravel()), axis=1)
        g_tag = bm.full((len(gp),), 3, dtype=bm.int64)

        r = bm.concatenate((fp, wp, dp, gp), axis=0)
        tag = bm.concatenate((f_tag, w_tag, d_tag, g_tag), axis=0)
        u = bm.zeros((len(r), 2), dtype=bm.float64)
        u = bm.set_at(u, (tag == 0)|(tag == 3), u_in)
        rho = bm.full((len(r),), rho_0, dtype=bm.float64)
        m0 = rho_0 * (init_domain[1] - init_domain[0]) * (init_domain[3] - init_domain[2]) / fp.shape[0]
        m1 = rho_0 * 6 * dx * (domain[3] - domain[2]) / gp.shape[0]
        mass0 = bm.full((len(r[tag != 3]),), m0, dtype=bm.float64)
        mass1 = bm.full((len(r[tag == 3]),), m1, dtype=bm.float64)
        mass = bm.concatenate((mass0, mass1), axis=0)
        
        nodedata = {
            "position": r,
            "tag": tag,
            "u": u,
            "dudt": bm.zeros_like(u),
            "rho": rho,
            "drhodt": bm.zeros_like(rho),
            "p": bm.zeros_like(rho),
            "sound": bm.zeros_like(rho),
            "mass": mass,
            "mu": bm.zeros_like(rho),
            "drdt": bm.zeros_like(r),
            "dx": dx,
        }
        return cls(r, nodedata=nodedata)

class Space:
    def raw_transform(self, box:Box, R:TensorLike):
        if box.ndim == 0 or box.size == 1:
            
            return R * box
        elif box.ndim == 1:
            indices = self._get_free_indices(R.ndim - 1) + "i"
            
            return bm.einsum(f"i,{indices}->{indices}", box, R)
        elif box.ndim == 2:
            free_indices = self._get_free_indices(R.ndim - 1)
            left_indices = free_indices + "j"
            right_indices = free_indices + "i"
            
            return bm.einsum(f"ij,{left_indices}->{right_indices}", box, R)
        raise ValueError(
            ("Box must be either: a scalar, a vector, or a matrix. " f"Found {box}.")
        )

    def _get_free_indices(self, n: int):
        
        return "".join([chr(ord("a") + i) for i in range(n)])

    def pairwise_displacement(self, Ra: TensorLike, Rb: TensorLike):
        if len(Ra.shape) != 1:
            msg = (
				"Can only compute displacements between vectors. To compute "
				"displacements between sets of vectors use vmap or TODO."
				)
            raise ValueError(msg)

        if Ra.shape != Rb.shape:
            msg = "Can only compute displacement between vectors of equal dimension."
            raise ValueError(msg)

        return Ra - Rb

    def periodic_displacement(self, side: Box, dR: TensorLike):
        _dR = ((dR + side * 0.5) % side) - 0.5 * side
        return _dR

    def periodic_shift(self, side: Box, R: TensorLike, dR: TensorLike):

        return (R + dR) % side

    def periodic(self, side: Box, wrapped: bool = True):
        def displacement_fn( Ra: TensorLike, Rb: TensorLike, perturbation = None, **unused_kwargs):
            if "box" in unused_kwargs:
                raise UnexpectedBoxException(
                    (
                        "`space.periodic` does not accept a box "
                        "argument. Perhaps you meant to use "
                        "`space.periodic_general`?"
                    )
                )
            dR = self.periodic_displacement(side, self.pairwise_displacement(Ra, Rb))
            if perturbation is not None:
                dR = self.raw_transform(perturbation, dR)
            
            return dR
        if wrapped:
            def shift_fn(R: TensorLike, dR: TensorLike, **unused_kwargs):
                if "box" in unused_kwargs:
                    raise UnexpectedBoxException(
                        (
                            "`space.periodic` does not accept a box "
                            "argument. Perhaps you meant to use "
                            "`space.periodic_general`?"
                        )
                    )

                return self.periodic_shift(side, R, dR)
        else:
                def shift_fn(R: TensorLike, dR: TensorLike, **unused_kwargs):
                    if "box" in unused_kwargs:
                        raise UnexpectedBoxException(
                            (
                                "`space.periodic` does not accept a box "
                                "argument. Perhaps you meant to use "
                                "`space.periodic_general`?"
                            )
                        )
                    return R + dR

        return displacement_fn, shift_fn

    def distance(self, dR: TensorLike):
        dr = self.square_distance(dR)
        return self.safe_mask(dr > 0, bm.sqrt, dr)

    def square_distance(self, dR: TensorLike):
        return bm.sum(dR**2, axis=-1)

    def safe_mask(self, mask, fn, operand, placeholder=0):
        masked = bm.where(mask, operand, 0)
        return bm.where(mask, fn(masked), placeholder)

class NeighborManager():
    @staticmethod
    def wall_virtual(position, tag):
        """Neighbor relationship between solid wall particles and virtual particles"""
        vir_r = position[tag == 2]
        wall_r = position[tag == 1]
        tree = cKDTree(wall_r)
        distance, neighbors = tree.query(vir_r, k=1)
        
        fuild_len = len(position[tag == 0]) 
        neighbors = neighbors + fuild_len
        node_self = bm.where(tag == 2)[0]
        return node_self, neighbors

    @staticmethod
    def fuild_fwvg(state, node_self, neighbors, dr_i_j, dist, w_dist, grad_w_dist):
        """Neighbor relations among fluid particles, solid wall particles, and virtual particles"""
        tag = state["tag"]
        wvg_tag = bm.where((tag == 1) | (tag == 2) | (tag == 3))[0]
        wvg_indx = bm.where(bm.isin(node_self, wvg_tag))[0] 
        wvg_mask = bm.ones(len(node_self), dtype=bm.bool)
        wvg_mask = bm.set_at(wvg_mask, wvg_indx, False)
        f_node = node_self[wvg_mask]
        neighbors = neighbors[wvg_mask]
        dr_i_j = dr_i_j[wvg_mask]
        dist = dist[wvg_mask]
        w_dist = w_dist[wvg_mask]
        grad_w_dist = grad_w_dist[wvg_mask]
        return f_node, neighbors, dr_i_j, dist, w_dist, grad_w_dist

    @staticmethod
    def wall_fg(state, node_self, neighbors, w_dist):
        """Neighbor relationship between fluid particles and solid wall particles"""
        tag = state["tag"]
        fvg_tag = bm.where((tag == 0) | (tag == 2) | (tag == 3))[0]
        fvg_indx = bm.where(bm.isin(node_self, fvg_tag))[0]
        fvg_mask = bm.ones(len(node_self), dtype=bm.bool)
        fvg_mask = bm.set_at(fvg_mask, fvg_indx, False)
        w_node = node_self[fvg_mask]
        fwvg_neighbors = neighbors[fvg_mask]
        w_dist = w_dist[fvg_mask]

        wv_tag = bm.where((tag == 1) | (tag == 2))[0]
        wv_indx = bm.where(bm.isin(fwvg_neighbors, wv_tag))[0]
        wv_mask = bm.ones(len(fwvg_neighbors), dtype=bm.bool)
        wv_mask = bm.set_at(wv_mask, wv_indx, False)
        w_node = w_node[wv_mask]
        fg_neighbors = fwvg_neighbors[wv_mask]
        w_dist = w_dist[wv_mask]
        return w_node, fg_neighbors, w_dist