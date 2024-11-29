from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import TriangleMesh, TetrahedronMesh, QuadrangleMesh

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator

from soptx.utils.timer import timer


bm.set_backend('numpy')
nx, ny = 10, 10
mesh_simplex = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
mesh_tensor = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)

p = 4

q = p+1
qf_simplex = mesh_simplex.quadrature_formula(q)
bcs_simplex, _ = qf_simplex.get_quadrature_points_and_weights()
phi_simplex = mesh_simplex.shape_function(bcs=bcs_simplex, p=p) # (NQ, ldof)

qf_tensor = mesh_tensor.quadrature_formula(q)
bcs_tensor, _ = qf_tensor.get_quadrature_points_and_weights()
phi_tensor = mesh_tensor.shape_function(bcs=bcs_tensor, p=p) # (NQ, ldof)

space = LagrangeFESpace(mesh_simplex, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
tldof = tensor_space.number_of_global_dofs()
print(f"tldof: {tldof}")
linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                        lame_lambda=1, shear_modulus=1, 
                                        hypo='plane_stress')
integrator0 = LinearElasticIntegrator(material=linear_elastic_material,
                                    q=tensor_space.p+3)
integrator1 = LinearElasticIntegrator(material=linear_elastic_material,
                                    q=tensor_space.p+3, method='fast_stress')
integrator2 = LinearElasticIntegrator(material=linear_elastic_material, 
                                    q=tensor_space.p+3, method='symbolic_stress')
integrator1.keep_data()
integrator2.keep_data()   # 保留中间数据
# integrator2.keep_result() # 保留积分结果
# 创建计时器
t = timer("Local Assembly Timing")
next(t)  # 启动计时器

KE0 = integrator0.assembly(space=tensor_space)
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