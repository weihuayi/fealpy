from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import HexahedronMesh, UniformMesh3d

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

class BoxDomainUnPolyloaded3d():
    def __init__(self):
        pass
        
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = 2*x**3 - 3*x*y**2 - 3*x*z**2
        val[..., 1] = 2*y**3 - 3*y*x**2 - 3*y*z**2
        val[..., 2] = 2*z**3 - 3*z*y**2 - 3*z*x**2
        
        return val

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)

def create_mesh(mesh_type, nx, ny, nz, h, origin=[0.0, 0.0, 0.0]):
    """根据参数创建不同类型的网格"""
    extent = [0, nx, 0, ny, 0, nz]
    
    if mesh_type == 'uniform':
        return UniformMesh3d(extent=extent, h=h, origin=origin,
                           ipoints_ordering='zyx', flip_direction=None, 
                           device='cpu')
    elif mesh_type == 'hexahedron':
        box = [0, nx*h[0], 0, ny*h[1], 0, nz*h[2]]  
        mesh = HexahedronMesh.from_box(box, nx=nx, ny=ny, nz=nz)
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
                        help='Specify the solver type for solving the linear system, default is "cg".')
    parser.add_argument('--nx', 
                        default=4, type=int, 
                        help='Initial number of grid cells in the x direction, default is 2.')
    parser.add_argument('--ny',
                        default=4, type=int,
                        help='Initial number of grid cells in the y direction, default is 2.')
    parser.add_argument('--nz',
                        default=4, type=int,
                        help='Initial number of grid cells in the z direction, default is 2.')
    parser.add_argument('--mesh-type',
                                choices=['uniform', 'hexahedron'],
                                default='hexahedron', type=str,
                                help='Type of mesh to use for computation.')
    args = parser.parse_args()

    pde = BoxDomainPolyloaded3d()
    args = parser.parse_args()

    bm.set_backend(args.backend)

    h = [1.0, 1.0, 1.0]
    mesh = create_mesh(args.mesh_type, args.nx, args.ny, args.nz, h)
    
    GD = mesh.geo_dimension()

    p = args.degree

    tmr = timer("FEM Solver")
    next(tmr)

    maxit = 4
    errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
    errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
    NDof = bm.zeros(maxit, dtype=bm.int32)
    for i in range(maxit):
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, 3))
        NDof[i] = tensor_space.number_of_global_dofs()

        linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                    lame_lambda=1, shear_modulus=1, 
                                                    hypo='3D', device=bm.get_device(mesh))
        tmr.send('material')

        integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                               q=tensor_space.p+1, method=None)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(integrator_K)
        K = bform.assembly(format='csr')
        # tmr.send('stiffness assembly')

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
        
        if isinstance(mesh, UniformMesh3d):
            mesh.to_vtk(
            '/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elastic/linear_elastic_examples/results/exp_3d_uh.vts')
        else:
            mesh.to_vtk(
            '/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elastic/linear_elastic_examples/results/exp_3d_uh.vtu')
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