"""
UniformMesh2d 下比较不同线弹性积分子矩阵组装方法的效率
"""
from fealpy.backend import backend_manager as bm
from fealpy.material import LinearElasticMaterial   
from fealpy.fem import LinearElasticIntegrator
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import HexahedronMesh, UniformMesh2d
from soptx.utils import timer
bm.set_backend('numpy')

t = timer("Assemble Timing")
next(t)  # 启动计时器

p = 1
n = 600
h = [1.0, 1.0]
origin = [0.0, 0.0]
mesh = UniformMesh2d(
            extent=[0, n, 0, n], h=h, origin=origin,
            ipoints_ordering='yx', flip_direction=None,
            device='cpu')
space= LagrangeFESpace(mesh, p=p)
tensor_space = TensorFunctionSpace(space, shape=(2, -1)) # dof_priority
tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
# tensor_space = TensorFunctionSpace(space, shape=(-1, 2)) # gd_priority
GD = mesh.geo_dimension()
ldof = space.number_of_local_dofs()
cm = mesh.entity_measure('cell')
q = p+1
qf = mesh.quadrature_formula(q)
bcs, ws = qf.get_quadrature_points_and_weights()
E = 1
nu = 0.3
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='plane_stress', device=bm.get_device(mesh))
integrator = LinearElasticIntegrator(material=linear_elastic_material, q=q)
integrator.keep_data(True)
integrator_vu = LinearElasticIntegrator(material=linear_elastic_material,
                                        q=q, method='voigt_uniform')
integrator_vu.keep_data(True)
integrator_fs = LinearElasticIntegrator(material=linear_elastic_material,
                                        q=q, method='fast_stress')
integrator_fs.keep_data(True)
integrator_fsu = LinearElasticIntegrator(material=linear_elastic_material, 
                                        q=q, method='fast_stress_uniform')
integrator_fsu.keep_data(True)
t.send('准备')

KE = integrator.assembly(tensor_space)
t.send('标准1')
KE_cache = integrator.assembly(tensor_space)
t.send('标准2')
KE_vu = integrator_vu.voigt_assembly_uniform(tensor_space)
t.send('voigt一致网格1')
KE_vu_cache = integrator_vu.voigt_assembly_uniform(tensor_space)
t.send('voigt一致网格2')
KE_fs = integrator_fs.fast_assembly_stress(tensor_space)
t.send('快速组装1')
KE_fs_cache = integrator_fs.fast_assembly_stress(tensor_space)
t.send('快速组装2')
KE_fsu = integrator_fsu.fast_assembly_stress_uniform(tensor_space)
t.send('快速组装一致网格1')
KE_fsu_cache = integrator_fsu.fast_assembly_stress_uniform(tensor_space)
t.send('快速组装一致网格2')
t.send(None)
error1 = bm.sum(bm.abs(KE[0] - KE_vu[0]))
error2 = bm.sum(bm.abs(KE[0] - KE_fs[0]))
error3 = bm.sum(bm.abs(KE[0] - KE_fsu[0]))
print(f"error1: {error1}")
print(f"error2: {error2}")
print(f"error3: {error3}")
print("----------------")
