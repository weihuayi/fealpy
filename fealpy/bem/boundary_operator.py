import numpy as np
from scipy.sparse import csr_matrix
from fealpy.mesh import IntervalMesh, TriangleMesh, QuadrangleMesh, UniformMesh3d
from fealpy.pde.bem_model_2d import PoissonModelConstantDirichletBC2d
from fealpy.pde.bem_model_3d import *
from fealpy.functionspace import LagrangeFESpace
from potential_grad_potential_integrator import PotentialGradPotentialIntegrator
from grad_potential_integrator import GradPotentialIntegrator
from scalar_source_integrator import ScalarSourceIntegrator
from matplotlib import pyplot as plt

def boundary_mesh_build(mesh):
    Boundary_Mesh = {
        "TriangleMesh": IntervalMesh,
        "QuadrangleMesh": IntervalMesh,
        "UniformMesh2d": IntervalMesh,
        "TetrahedronMesh": TriangleMesh,
        "HexahedronMesh": QuadrangleMesh,
        "UniformMesh3d": QuadrangleMesh,
    }
    if type(mesh).__name__ == "UniformMesh3d":
        bd_face = mesh.ds.boundary_face()[:, [0, 2, 3, 1]]
    else:
        bd_face = mesh.ds.boundary_face()
    node = mesh.entity('node')
    old_bd_node_idx = mesh.ds.boundary_node_index()
    new_node = node[old_bd_node_idx]
    aux_idx1 = np.zeros(len(node), dtype=np.int_)
    aux_idx2 = np.arange(len(old_bd_node_idx), dtype=np.int_)
    aux_idx1[old_bd_node_idx] = aux_idx2
    new_cell = aux_idx1[bd_face]
    bd_mesh = Boundary_Mesh[type(mesh).__name__](new_node, new_cell)

    return bd_mesh


def error_calculator(mesh, u, v, q=3, power=2):
    qf = mesh.integrator(q, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs)

    cell = mesh.entity('cell')
    cell_node_val = u[cell]
    if type(mesh).__name__ == "UniformMesh3d":
        bc0 = bcs[0].reshape(-1, 2)  # (NQ0, 2)
        bc1 = bcs[1].reshape(-1, 2)  # (NQ1, 2)
        bc2 = bcs[2].reshape(-1, 2)  # (NQ2, 2)
        bc = np.einsum('im, jn, kl->ijkmnl', bc0, bc1, bc2).reshape(-1, 8)  # (NQ0, NQ1, NQ2, 2, 2, 2)  (NQ0*NQ1*NQ2, 8)

        u = np.einsum('...j, cj->...c', bc, cell_node_val)
    else:
        u = np.einsum('...j, cj->...c', bcs, cell_node_val)
    if callable(v):
        if not hasattr(v, 'coordtype'):
            v = v(ps)
        else:
            if v.coordtype == 'cartesian':
                v = v(ps)
            elif v.coordtype == 'barycentric':
                v = v(bcs)

    if u.shape[-1] == 1:
        u = u[..., 0]

    if v.shape[-1] == 1:
        v = v[..., 0]

    cm = mesh.entity_measure('cell')

    f = np.power(np.abs(u - v), power)

    e = np.einsum('q, qc..., c->c...', ws, f, cm)
    e = np.power(np.sum(e), 1 / power)

    return e

class BoundaryOperator:

    def __init__(self, space):
        self.space = space
        self._H = None
        self._G = None
        self.dintegrators = []  # 区域积分子
        self.bintegrators = []  # 边界积分子


    def add_domain_integrator(self, I):
        """
        @brief 增加一个或多个区域积分对象
        """
        if isinstance(I, list):
            self.dintegrators.extend(I)
        else:
            self.dintegrators.append(I)

    def add_boundary_integrator(self, I):
        """
        @brief 增加一个或多个边界积分对象
        """
        if isinstance(I, list):
            self.bintegrators.extend(I)
        else:
            self.bintegrators.append(I)

    def assembly(self):
        """
        @brief 数值积分组装

        @note space 可能是以下的情形, 程序上需要更好的设计
            * 标量空间
            * 由标量空间组成的向量空间
            * 由标量空间组成的张量空间
            * 向量空间（基函数是向量型的）
            * 张量空间（基函数是张量型的
        """
        if isinstance(self.space, tuple) and not isinstance(self.space[0], tuple):
            # 由标量函数空间张成的向量函数空间
            return self.assembly_for_vspace_with_scalar_basis()
        else:
            # 标量函数空间或基是向量函数的向量函数空间
            return self.assembly_for_sspace_and_vspace_with_vector_basis()

    def assembly_for_sspace_and_vspace_with_vector_basis(self):
        # ===================================================
        space = self.space
        if space.p == 0:
            gdof = space.mesh.number_of_cells()
        else:
            gdof = space.dof.number_of_global_dofs()

        Hij, Gij = self.bintegrators[0].assembly_face_matrix(space)

        face2dof = space.dof.cell_to_dof()
        I = np.broadcast_to(np.arange(gdof, dtype=np.int64)[:, None, None], shape=Hij.shape)
        J = np.broadcast_to(face2dof[None, ...], shape=Hij.shape)

        # 整体矩阵的初始化与组装
        self._H = np.zeros((gdof, gdof))
        np.add.at(self._H, (I, J), Hij)
        np.fill_diagonal(self._H, 0.5)
        self._G = np.zeros((gdof, gdof))
        np.add.at(self._G, (I, J), Gij)
        bd_face_measure = space.mesh.entity_measure('cell')
        # TODO: 补充高次与高维情况下，对奇异积分的处理
        if space.GD == 2:
            np.fill_diagonal(self._G, (bd_face_measure * (np.log(2 / bd_face_measure) + 1) / np.pi / 2))
        # ===================================================
        f = self.dintegrators[0].assembly_cell_vector(space)
        self._f = f

        return self._H, self._G, self._f

    def assembly_for_vspace_with_scalar_basis(self):

        raise NotImplementedError

    def update(self):
        """
        @brief 当空间改变时，重新组装向量
        """
        return self.assembly()


if __name__ == '__main__':
    from fealpy.bem.boundary_condition import DirichletBC
    from fealpy.bem.internal_operator import InternalOperator
    from fealpy.tools.show import showmultirate
    pde = PoissonModelConstantDirichletBC2d()
    box = pde.domain()
    nx = 5
    ny = 5
    # 定义网格对象
    mesh = TriangleMesh.from_box(box, nx, ny)
    # pde = PoissonModelConstantDirichletBC3d()
    # nx = 3
    # ny = 3
    # nz = 3
    #
    # hx = (1 - 0) / nx
    # hy = (1 - 0) / ny
    # hz = (1 - 0) / nz
    # mesh = UniformMesh3d((0, nx, 0, ny, 0, nz), h=(hx, hy, hz), origin=(0, 0, 0))  #
    # mesh.to_vtk_file(filename='ori_quad.vtu')
    # 构造边界网格与空间
    p = 1
    maxite = 3
    errorMatrix = np.zeros(maxite)
    N = np.zeros(maxite)

    for k in range(maxite):
        bd_mesh = boundary_mesh_build(mesh)
        # bd_mesh.to_vtk(fname='test_quad.vtu')
        space = LagrangeFESpace(bd_mesh, p=p)
        space.domain_mesh = mesh

        bd_operator = BoundaryOperator(space)
        bd_operator.add_boundary_integrator(PotentialGradPotentialIntegrator(q=2))
        bd_operator.add_domain_integrator(ScalarSourceIntegrator(f=pde.source, q=3))

        H, G, F = bd_operator.assembly()
        bc = DirichletBC(space=space, gD=pde.dirichlet)
        G, F = bc.apply(H, G, F)
        xi = space.xi
        u = pde.dirichlet(xi)
        q = np.linalg.solve(G, F)

        inter_operator = InternalOperator(space)
        inter_operator.add_boundary_integrator(PotentialGradPotentialIntegrator(q=2))
        inter_operator.add_domain_integrator(ScalarSourceIntegrator(f=pde.source, q=3))
        inter_H, inter_G, inter_F = inter_operator.assembly()
        u_inter = inter_G@q - inter_H@u + inter_F

        result_u = np.zeros(mesh.number_of_nodes())
        result_u[mesh.ds.boundary_node_flag()] = pde.dirichlet(mesh.entity('node')[mesh.ds.boundary_node_flag()])
        result_u[~mesh.ds.boundary_node_flag()] = u_inter

        errorMatrix[k] = error_calculator(mesh, result_u, pde.solution)

        v = mesh.entity_measure('cell')
        h = np.max(v)
        N[k] = np.power(h, 1 / mesh.geo_dimension())

        if k < maxite:
            mesh.uniform_refine(1)

    print(f'迭代{maxite}次，结果如下：')
    print("误差：\n", errorMatrix)
    print('误差比：\n', errorMatrix[0:-1] / errorMatrix[1:])

    fig = plt.figure()
    axes = showmultirate(plt, 0, N[None, ...], errorMatrix[None, ...], labellist=[['l2']])
    plt.show()






