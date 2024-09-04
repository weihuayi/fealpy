from fealpy.experimental.backend import backend_manager as bm
# bm.set_backend('numpy')
# bm.set_backend('pytorch')
bm.set_backend('jax')

from fealpy.experimental.mesh import UniformMesh2d

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator

from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

extent = [0, 2, 0, 2]
h = [1, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)

p = 2
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(2, -1))

# (tgdof, )
uh_dependent = tensor_space.function()

# 与单元有关的组装方法
integrator_bi_dependent = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                        elasticity_type='strain', q=5)

# 与单元有关的组装方法
KK_dependent = integrator_bi_dependent.assembly(space=tensor_space)

# 与单元有关的组装方法 
bform_dependent = BilinearForm(tensor_space)
bform_dependent.add_integrator(integrator_bi_dependent)
K_dependent = bform_dependent.assembly()
