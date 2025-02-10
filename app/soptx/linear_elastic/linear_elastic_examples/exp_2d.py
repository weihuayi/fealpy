from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import UniformMesh2d, TriangleMesh, QuadrangleMesh

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC

from fealpy.decorator import cartesian
from fealpy.solver import cg, spsolve

from app.soptx.soptx.utils.timer import timer

import argparse

class BoxDomainData2d:
    """
    @brief Dirichlet 边界条件的线弹性问题模型
    @note 本模型假设在二维方形区域 [0,1] x [0,1] 内的线性弹性问题
    """
    def __init__(self, E=1.0, nu=0.3):
        """
        @brief 构造函数
        @param[in] E 弹性模量，默认值为 1.0
        @param[in] nu 泊松比，默认值为 0.3
        """
        self.E = E 
        self.nu = nu

        self.lam = self.nu * self.E / ((1 + self.nu) * (1 - 2*self.nu))
        self.mu = self.E / (2 * (1+self.nu))

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, p):
        """
        @brief 模型的源项值 f
        """
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
        val[..., 1] = -25/26*(-1+2*y) * (-1+2*x)

        return val

    @cartesian
    def solution(self, p):
        """
        @brief 模型真解
        """
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros(p.shape, dtype=bm.float64)
        val[..., 0] = x*(1-x)*y*(1-y)
        val[..., 1] = 0

        return val

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief 判断给定点是否在 Dirichlet 边界上
        @param[in] p 一个表示空间点坐标的数组
        @return 如果在 Dirichlet 边界上，返回 True，否则返回 False
        """
        x = p[..., 0]
        y = p[..., 1]
        flag1 = bm.abs(x) < 1e-13
        flag2 = bm.abs(x - 1) < 1e-13
        flagx = bm.logical_or(flag1, flag2)
        flag3 = bm.abs(y) < 1e-13
        flag4 = bm.abs(y - 1) < 1e-13
        flagy = bm.logical_or(flag3, flag4)
        flag = bm.logical_or(flagx, flagy)

        return flag

def create_mesh(mesh_type, nx, ny, h, origin=[0.0, 0.0]):
    """根据参数创建不同类型的网格"""
    extent = [0, nx, 0, ny]
    
    if mesh_type == 'uniform':
        return UniformMesh2d(extent=extent, h=h, origin=origin,
                           ipoints_ordering='yx', flip_direction=None, device='cpu')
    elif mesh_type == 'triangle':
        box = [0, nx*h[0], 0, ny*h[1]]  
        mesh = TriangleMesh.from_box(box, nx=nx, ny=ny)
        return mesh
    elif mesh_type == 'quadrangle':
        box = [0, nx*h[0], 0, ny*h[1]]  
        mesh = QuadrangleMesh.from_box(box, nx=nx, ny=ny)
        return mesh
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}")
    
def main():
    parser = argparse.ArgumentParser(description="Solve linear elasticity problems \
                                in arbitrary order Lagrange finite element space on HexahedronMesh.")
    parser.add_argument('--backend',
                        choices=['numpy', 'pytorch'], 
                        default='numpy', type=str,
                        help='Specify the backend type for computation, default is "pytorch".')
    parser.add_argument('--degree', 
                        default=1, type=int, 
                        help='Degree of the Lagrange finite element space, default is 1.')
    parser.add_argument('--solver',
                        choices=['cg', 'spsolve'],
                        default='spsolve', type=str,
                        help='Specify the solver type for solving the linear system, default is "mumps".')
    parser.add_argument('--nx', 
                        default=10, type=int, 
                        help='Initial number of grid cells in the x direction, default is 10.')
    parser.add_argument('--ny',
                        default=10, type=int,
                        help='Initial number of grid cells in the y direction, default is 10.')
    parser.add_argument('--mesh-type',
                            choices=['uniform', 'triangle', 'quadrangle'],
                            default='uniform', type=str,
                            help='Type of mesh to use for computation.')
    args = parser.parse_args()

    pde = BoxDomainData2d()
    args = parser.parse_args()

    bm.set_backend(args.backend)

    h = [1.0, 1.0]
    mesh = create_mesh(args.mesh_type, args.nx, args.ny, h)
    GD = mesh.geo_dimension()

    p = args.degree

    tmr = timer(f"Solver with {args.mesh_type} and {args.solver}")
    next(tmr)

    maxit = 4
    errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
    errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
    NDof = bm.zeros(maxit, dtype=bm.int32)
    for i in range(maxit):
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
        NDof[i] = tensor_space.number_of_global_dofs()

        linear_elastic_material = LinearElasticMaterial(name='E1nu03', 
                                                    elastic_modulus=1, poisson_ratio=0.3, 
                                                    hypo='plane_strain', device=bm.get_device(mesh))
        tmr.send('material')

        integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                            q=tensor_space.p+1, method=None)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(integrator_K)
        K = bform.assembly(format='csr')
        tmr.send('stiffness assembly')

        integrator_F = VectorSourceIntegrator(source=pde.source, 
                                            q=tensor_space.p+1)
        lform = LinearForm(tensor_space)    
        lform.add_integrator(integrator_F)
        F = lform.assembly()
        tmr.send('source assembly')

        dbc = DirichletBC(space=tensor_space, 
                        gd=pde.dirichlet, 
                        threshold=None, 
                        method='interp')
        # K, F = dbc.apply(A=K, f=F, uh=None, gd=pde.dirichlet, check=True)
        uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
                        dtype=bm.float64, device=bm.get_device(mesh))
        uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, uh=uh_bd, 
                                                        threshold=None, method='interp')
        F = F - K.matmul(uh_bd)
        F = bm.set_at(F, isDDof, uh_bd[isDDof])
        K = dbc.apply_matrix(matrix=K, check=True)
        tmr.send('boundary')
        
        uh = tensor_space.function()

        if args.solver == 'cg':
            uh[:] = cg(K, F, maxiter=1000, atol=1e-14, rtol=1e-14)
        elif args.solver == 'spsolve':
            uh[:] = spsolve(K, F, solver='mumps')
        # t.send('Solving Phase')  # 记录求解阶段时间

        NN = mesh.number_of_nodes()
        if tensor_space.dof_priority:
            uh_show = uh.reshape(GD, NN).T
        else:
            uh_show = uh.reshape(NN, GD)
        mesh.nodedata['uh'] = uh_show[:]
        
        if isinstance(mesh, UniformMesh2d):
            mesh.to_vtk(
            '/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elastic/linear_elastic_examples/results/exp_2d_uh.vts')
        else:
            mesh.to_vtk(
            '/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elastic/linear_elastic_examples/results/exp_2d_uh.vtk')
        tmr.send('solve({})'.format(args.solver))
        
        tmr.send(None)

        u_exact = tensor_space.interpolate(pde.solution)
        errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / NDof[i])))
        errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)

        if i < maxit-1:
            mesh.uniform_refine()

    print("errorMatrix:\n", errorType, "\n", errorMatrix)
    print("NDof:", NDof)
    print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
    print("order_L2:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))

if __name__ == "__main__":
    main()