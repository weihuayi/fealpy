from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import (TriangleMesh, TetrahedronMesh, 
                         QuadrangleMesh, HexahedronMesh, 
                         UniformMesh2d, UniformMesh3d)

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator

from soptx.utils.timer import timer


bm.set_backend('numpy')
box = [0, 1, 0, 1]
nx, ny = 10, 10
extent = [0, nx, 0, ny]
h = [(box[1] - box[0])/nx, (box[3] - box[2])/ny]
origin = [0, 0]
mesh_simplex = TriangleMesh.from_box(box=box, nx=nx, ny=ny)
mesh_tensor = QuadrangleMesh.from_box(box=box, nx=nx, ny=ny)
mesh_struct = UniformMesh2d(extent=extent, h=h, origin=origin)

p = 1
space_simplex = LagrangeFESpace(mesh_simplex, p=p, ctype='C')
ldof1 = space_simplex.number_of_local_dofs()
space_tensor = LagrangeFESpace(mesh_tensor, p=p, ctype='C')
tensor_space_simplex = TensorFunctionSpace(space_simplex, shape=(-1, 2))
tldof1 = tensor_space_simplex.number_of_local_dofs()
tensor_space_tensor = TensorFunctionSpace(space_tensor, shape=(-1, 2))

linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='plane_stress')

integrator_standard = LinearElasticIntegrator(material=linear_elastic_material, 
                                                q=p+3)
integrator_fast_stress = LinearElasticIntegrator(material=linear_elastic_material, 
                                                q=p+3, method='fast_stress')
integrator_symbolic_stress = LinearElasticIntegrator(material=linear_elastic_material, 
                                                q=p+3, method='symbolic_stress')
# 保留中间数据
integrator_standard.keep_data()
integrator_fast_stress.keep_data()     
integrator_symbolic_stress.keep_data() 
# integrator2.keep_result() # 保留积分结果

# 创建计时器
t = timer("2d Local Assembly Timing (Simplex)")
next(t)  # 启动计时器
KE_symbolic_stress_simplex = integrator_symbolic_stress.symbolic_assembly_stress(space=tensor_space_simplex)
t.send('Symbolic Assembly simplex')
KE_symbolic_stress_simplex_cache = integrator_symbolic_stress.symbolic_assembly_stress(space=tensor_space_simplex)
t.send('Symbolic Assembly simplex cache')


KE_standard_simplex = integrator_standard.assembly(space=tensor_space_simplex)
t.send('Standard Assembly simplex')
KE_standard_simplex_cache = integrator_standard.assembly(space=tensor_space_simplex)
t.send('Standard Assembly simplex cache')
KE_fast_stress_simplex = integrator_fast_stress.fast_assembly_stress(space=tensor_space_simplex)
t.send('Fast Assembly simplex')
KE_fast_stress_simplex_cache = integrator_fast_stress.fast_assembly_stress(space=tensor_space_simplex)
t.send('Fast Assembly simplex cache')

# t.send(None)

t = timer("2d Local Assembly Timing (Tensor)")
next(t)  # 启动计时器

KE_standard_tensor = integrator_standard.assembly(space=tensor_space_tensor)
t.send('Standard Assembly tensor')
KE_standard_tensor_cache = integrator_standard.assembly(space=tensor_space_tensor)
t.send('Standard Assembly tensor cache')
KE_fast_stress_tensor = integrator_fast_stress.fast_assembly_stress(space=tensor_space_tensor)
t.send('Fast Assembly tensor')
KE_fast_stress_tensor_cache = integrator_fast_stress.fast_assembly_stress(space=tensor_space_tensor)
t.send('Fast Assembly tensor cache')

error1 = bm.sum(bm.abs(KE_standard_tensor - KE_fast_stress_tensor))
print(f"error: {error1}")

t.send(None)

print("-------------------------------")