from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm

class CouplingInterface:
    """FSI coupling operator driven by interface triangles.

    - Pressure on fluid interface -> nodal force on solid (traction -p n, lumped to vertices).
    - Solid displacement (NN_solid, GD) -> fluid interface displacement / velocity.

    Use :meth:`from_volume_mesh` with ``node`` and ``interface_tri`` from
    :meth:`fealpy.mesher.gmsh_fsi_pipe_mesher.BaseGmshFSIPipeMesher.extract_mesh_data`,
    or pass two :class:`TriangleMesh` instances (``__init__``).
    """

    def __init__(self, pde):
        self.pde = pde

    def interface_normal(self):
        '''
            计算外法向量
        '''
        pde = self.pde
        tri_interface = pde.interface_mesh
        node = tri_interface.node
        cell = tri_interface.cell
        v0 = node[cell[:, 1], :] - node[cell[:, 0], :]
        v1 = node[cell[:, 2], :] - node[cell[:, 0], :]
        nv = bm.cross(v0, v1)
        S = bm.sqrt(bm.sum(nv**2, axis=1))/2
        # 单元中心处法向量
        nv = nv / bm.sqrt(bm.sum(nv**2, axis=1))[:, None]

        n2c = tri_interface.node_to_cell()
        # 根据点附近单元面积计算单元对点的权重
        ws = bm.ones(n2c.shape)
        ws *= S
        ws = n2c.mul(ws)
        ws = ws.toarray()
        ws_sum = bm.sum(ws, axis=1)
        ws = ws / ws_sum[:, None]
        # 节点处法向量（加权平均），指向管道内部
        nv = -ws @ nv
        return nv

    def pressure_on_interface(self, p1: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        pde = self.pde
        is_wall = pde.is_wall_boundary(pde.fluid_mesh.entity('node'))
        p = p1[is_wall]
        pressurespace = LagrangeFESpace(mesh=pde.interface_mesh, p=1)
        pressure = pressurespace.function()
        pressure[:] = p

        pressspace = TensorFunctionSpace(pressurespace, (3, -1))
        press = pressspace.function()

        nv = self.interface_normal()
        press[:] = (pressure[:, None] * nv).T.reshape(-1)

        return press
    
    def pressure_on_solid(self, press: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

        pde = self.pde
        is_inwall = pde.is_wall_boundary(pde.solid_mesh.node)
        space = LagrangeFESpace(mesh=pde.solid_mesh, p=1)
        solid_pspace = TensorFunctionSpace(space, (3, -1))
        solid_p = solid_pspace.function()
        solid_p.reshape(3, -1)[:, is_inwall] = press.reshape(3, -1)

        return solid_p
    
    def shear_stress_on_interface(self, u1: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        pde = self.pde
        fluid_mesh = pde.fluid_mesh
        qf = fluid_mesh.quadrature_formula(q=4, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        uspace = u1.space
        grad_u = uspace.grad_value(uh = u1, bc = bcs)
        grad_u = bm.einsum("n, knij -> kij", ws, grad_u)
        grad_u_T = grad_u.transpose(0, 2, 1)
        cellmeasure = fluid_mesh.entity_measure("cell")
        n2c = fluid_mesh.node_to_cell()
        w = bm.ones(n2c.shape)
        w *= cellmeasure
        w = n2c.mul(w)
        w = w.toarray()
        w_sum = bm.sum(w, axis=1)
        w = w / w_sum[:, None]
        grad_u = bm.einsum("lk, kij -> lij", w, grad_u)
        grad_u_T = bm.einsum("lk, kij -> lij", w, grad_u_T)

        is_wall = pde.is_wall_boundary(pde.fluid_mesh.node)
        grad_u = grad_u[is_wall, :, :]
        grad_u_T = grad_u_T[is_wall, :, :]
        tau = grad_u + grad_u_T
        tau *= pde.mu
        nv = self.interface_normal()
        stress = bm.einsum("kij, kj -> ki", tau, nv)

        space = LagrangeFESpace(mesh=pde.interface_mesh, p=1)
        shear_stressspace = TensorFunctionSpace(space, (3, -1))
        shear_stress = shear_stressspace.function()
        shear_stress.reshape(3, -1).T[:] = stress
        return shear_stress
        
    def shear_stress_on_solid(self, shear_stress: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

        pde = self.pde
        is_inwall = pde.is_wall_boundary(pde.solid_mesh.node)
        space = LagrangeFESpace(mesh=pde.solid_mesh, p=1)
        solid_stressspace = TensorFunctionSpace(space, (3, -1))
        solid_stress = solid_stressspace.function()
        solid_stress.reshape(3, -1)[:, is_inwall] = shear_stress.reshape(3, -1)

        return solid_stress
    
    def structural_stress_on_interface(self, uh: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        pde = self.pde
        E = pde.E
        nu = pde.nu
        solid_mesh = pde.solid_mesh
        qf = solid_mesh.quadrature_formula(q=5, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad_u = uh.space.grad_value(uh = uh, bc = bcs)
        grad_u = bm.einsum("n, knij -> kij", ws, grad_u)
        grad_u_T = grad_u.transpose(0, 2, 1)
        cellmeasure = solid_mesh.entity_measure("cell")
        n2c = solid_mesh.node_to_cell()
        w = bm.ones(n2c.shape)
        w *= cellmeasure
        w = n2c.mul(w)
        w = w.toarray()
        w_sum = bm.sum(w, axis=1)
        w = w / w_sum[:, None]
        grad_u = bm.einsum("lk, kij -> lij", w, grad_u)
        grad_u_T = bm.einsum("lk, kij -> lij", w, grad_u_T)

        is_wall = pde.is_wall_boundary(pde.solid_mesh.node)
        grad_u = grad_u[is_wall, :, :]
        grad_u_T = grad_u_T[is_wall, :, :]

        epsilon = 0.5 * (grad_u + grad_u_T)
        epsilon_kk = bm.einsum("lii -> l", epsilon)
        G = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        I = bm.eye(3) 
        stress = 2 * G * epsilon
        stress += lam * (epsilon_kk[:, None, None] * I)

        nv = self.interface_normal()
        stress = bm.einsum("kij, kj -> ki", stress, nv)

        space = LagrangeFESpace(mesh=pde.interface_mesh, p=1)
        stressspace = TensorFunctionSpace(space, (3, -1))
        structural_stress = stressspace.function()
        structural_stress.reshape(3, -1).T[:] = stress
        return structural_stress
        # return stress


    