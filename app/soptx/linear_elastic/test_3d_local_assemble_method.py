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
box_2d = [0, 1, 0, 1]
box_3d = [0, 1, 0, 1, 0, 1]
nx, ny, nz = 4, 4, 4
extent_2d = [0, nx, 0, ny]
extent_3d = [0, nx, 0, ny, 0, nz]
h_2d = [(box_2d[1]-box_2d[0])/nx, (box_2d[3]-box_2d[2])/ny]
h_3d = [(box_3d[1]-box_3d[0])/nx, (box_3d[3]-box_3d[2])/ny, (box_3d[5]-box_3d[4])/nz]
origin_2d = [0, 0]
origin_3d = [0, 0, 0]
mesh_2d_simplex = TriangleMesh.from_box(box=box_2d, nx=nx, ny=ny)
mesh_3d_simplex = TetrahedronMesh.from_box(box=box_3d, nx=nx, ny=ny, nz=nx)
mesh_2d_tensor = QuadrangleMesh.from_box(box=box_2d, nx=nx, ny=ny)
mesh_3d_tensor = HexahedronMesh.from_box(box=box_3d, nx=nx, ny=ny, nz=nx)
mesh_2d_struct = UniformMesh2d(extent=extent_2d, h=h_2d, origin=origin_2d)
mesh_3d_struct = UniformMesh3d(extent=extent_3d, h=h_3d, origin=origin_3d)

p = 4
space_2d_simplex = LagrangeFESpace(mesh_2d_simplex, p=p, ctype='C')
space_3d_simplex = LagrangeFESpace(mesh_3d_simplex, p=p, ctype='C')
tensor_space_2d_simplex = TensorFunctionSpace(space_2d_simplex, shape=(-1, 2))
tensor_space_3d_simplex = TensorFunctionSpace(space_3d_simplex, shape=(-1, 3))

linear_elastic_material_2d = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='plane_stress')
linear_elastic_material_3d = LinearElasticMaterial(name='lam1_mu1',
                                                lame_lambda=1, shear_modulus=1,
                                                hypo='3D')

integrator_standard_2d = LinearElasticIntegrator(material=linear_elastic_material_2d, 
                                                q = p+3)
integrator_standard_3d = LinearElasticIntegrator(material=linear_elastic_material_3d, 
                                                q = p+3)
integrator_fast_stress_2d = LinearElasticIntegrator(material=linear_elastic_material_2d,
                                                q = p+3, method = 'fast_stress')
integrator_fast_stress_3d = LinearElasticIntegrator(material=linear_elastic_material_3d,
                                                q = p+3, method = 'fast_stress')
integrator_symbolic_stress_2d = LinearElasticIntegrator(material=linear_elastic_material_2d, 
                                                q = p+3, method = 'symbolic_stress')
integrator_symbolic_stress_3d = LinearElasticIntegrator(material=linear_elastic_material_3d,
                                                q = p+3, method = 'symbolic_stress')

integrator_standard_2d.keep_data()
integrator_standard_3d.keep_data()   # 保留中间数据
# integrator2.keep_result() # 保留积分结果

# 创建计时器
t = timer("2d Local Assembly Timing")
next(t)  # 启动计时器

KE_2d_simplex_standard = integrator_standard_2d.assembly(space=tensor_space_2d_simplex)
t.send('Standard Assembly 2d simplex')
KE_2d_simplex_standard



KE_2d_fast1 = integrator1.fast_assembly_stress(space=tensor_space)
t.send('Fast Assembly1')
KE12 = integrator1.fast_assembly_stress(space=tensor_space)
t.send('Fast Assembly2')

KE21 = integrator2.symbolic_assembly_stress(space=tensor_space)
t.send('Symbolic Assembly1')
KE22 = integrator2.symbolic_assembly_stress(space=tensor_space)
t.send('Symbolic Assembly2')
KE23 = integrator2.symbolic_assembly_stress(space=tensor_space)
t.send('Symbolic Assembly3')
# 结束计时
t.send(None)

t = timer("3d Local Assembly Timing")
next(t)  # 启动计时器

KE_2d_standard = integrator_standard.assembly(space=tensor_space)
t.send('Assembly')

KE11 = integrator1.fast_assembly_stress(space=tensor_space)
t.send('Fast Assembly1')
KE12 = integrator1.fast_assembly_stress(space=tensor_space)
t.send('Fast Assembly2')

KE21 = integrator2.symbolic_assembly_stress(space=tensor_space)
t.send('Symbolic Assembly1')
KE22 = integrator2.symbolic_assembly_stress(space=tensor_space)
t.send('Symbolic Assembly2')
KE23 = integrator2.symbolic_assembly_stress(space=tensor_space)
t.send('Symbolic Assembly3')
# 结束计时
t.send(None)

print("-------------------------------")