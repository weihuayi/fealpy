from fealpy.experimental.backend import backend_manager as bm
bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import TriangleMesh

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator

from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.typing import TensorLike, _S, Index


from fealpy.decorator import barycentric

@barycentric
def gd(bcs: TensorLike, index: Index=_S) -> TensorLike:
    val = (1 - bcs) ** 2
    return val

nx = 4
ny = 4
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
NC = mesh.number_of_cells()
p = 1
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(2, -1))

maxit = 5
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
for i in range(maxit):

    # 与单元有关的组装方法
    integrator_bi_dependent_none = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='strain', coef=None, q=p+3)
    KK_dependent_none = integrator_bi_dependent_none.assembly(space=tensor_space)
    KK_dependent_none_0 = KK_dependent_none[0]
    # 与单元无关的组装方法
    integrator_bi_independent_none = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            method='fast_strain', coef=None, q=p+3)
    KK_independent_none = integrator_bi_dependent_none.fast_assembly_strain_constant(space=tensor_space)
    KK_independent_none_0 = KK_independent_none[0]

    # 与单元有关的组装方法
    uh = tensor_space.function() # (tgdof, )
    @barycentric
    def gd(bcs: TensorLike, index: Index=_S) -> TensorLike:
        val = (0.5 - uh(bcs)) ** 2
        return val

    integrator_bi_dependent = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='strain', coef=gd, q=p+3)
    KK_dependent = integrator_bi_dependent.assembly(space=tensor_space)
    KK_dependent_0 = KK_dependent[0]

    errorMatrix[0, i] = bm.max(bm.abs(KK_dependent_none*0.25 - KK_dependent))
    errorMatrix[1, i] = bm.max(bm.abs(KK_independent_none*0.25 - KK_dependent))
    if i < maxit-1:
        mesh.uniform_refine()


print("errorMatrix:\n", errorMatrix)
