from fealpy.backend import backend_manager as bm
from fealpy.material import LinearElasticMaterial
from fealpy.fem import LinearElasticIntegrator
from fealpy.mesh import UniformMesh3d, TetrahedronMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.utils import timer

class LinearElasticLFEMModel:
    def __init__(self, mesh):
        bm.set_backend('numpy')
        self.linear_elastic_material = LinearElasticMaterial(name='test', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='3D')
        space = LagrangeFESpace(mesh, p=1)
        self.tensor_space = TensorFunctionSpace(space, shape=(3, -1))

    def assemble_exact(self):
        integrator_standard = LinearElasticIntegrator(
                                            material=self.linear_elastic_material, 
                                            q=5, method=None
                                        )
        KE_standard = integrator_standard.assembly(self.tensor_space)
        integrator_fast = LinearElasticIntegrator(
                                            material=self.linear_elastic_material, 
                                            q=5, method='fast'
                                        )
        KE_fast = integrator_fast.fast_assembly(self.tensor_space)
        diff1 = bm.sum(bm.abs(KE_standard - KE_fast))
        print(f"Difference between standard and fast methods: {diff1}")
        integrator_symbolic = LinearElasticIntegrator(
                                            material=self.linear_elastic_material, 
                                            q=5, method='symbolic'
                                        )
        KE_symbolic = integrator_symbolic.symbolic_assembly(self.tensor_space)
        diff2 = bm.sum(bm.abs(KE_standard - KE_symbolic))
        print(f"Difference between standard and symbolic methods: {diff2}")

    def assemble_time(self):
        integrator_standard = LinearElasticIntegrator(
                                    material=self.linear_elastic_material, 
                                    q=5, method=None
                                )
        integrator_standard.keep_data()
        integrator_fast = LinearElasticIntegrator(
                                    material=self.linear_elastic_material, 
                                    q=5, method='fast'
                                )
        integrator_fast.keep_data()
        integrator_symbolic = LinearElasticIntegrator(
                                    material=self.linear_elastic_material, 
                                    q=5, method='symbolic'
                                )
        integrator_symbolic.keep_data()
        for i in range(5):
            t = timer()
            next(t)  # 启动计时器
            KE_standard = integrator_standard.assembly(self.tensor_space)
            t.send('standard')
            KE_fast = integrator_fast.fast_assembly(self.tensor_space)
            t.send('fast')
            KE_symbolic = integrator_symbolic.symbolic_assembly(self.tensor_space)
            t.send('symbolic')
            t.send(None)
    
if __name__ == "__main__":
    nx, ny, nz = 5, 5, 5
    mesh_tet = TetrahedronMesh.from_box(
                                        box=[0, 10, 0, 10, 0, 10], 
                                        nx=10, ny=10, nz=10, 
                                        device='cpu')
    mesh_u3 = UniformMesh3d(
                        extent=[0, nx, 0, ny, 0, nz], h=[1, 1, 1], origin=[0, 0, 0],
                        ipoints_ordering='zyx', device='cpu'
                    )
    mesh_hex = HexahedronMesh.from_box(
                                    box=[0, 10, 0, 10, 0, 10], 
                                    nx=nx, ny=ny, nz=nz, 
                                    device='cpu')
    model = LinearElasticLFEMModel(mesh=mesh_u3)
    
    print(f"===== Testing assembly =====")
    # model.assemble_exact()
    model.assemble_time()
        
