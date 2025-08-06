from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm
from fealpy.fem import DirichletBC

from fealpy.csm.model.beam.timoshenko_beam_data_3d import TimoshenkoBeamData3D
from fealpy.csm.material.timoshenko_beam_material import TimoshenkoBeamMaterial
from fealpy.csm.fem.timoshenko_beam_integrator import TimoshenkoBeamIntegrator

# Example usage
model = TimoshenkoBeamData3D()
material = TimoshenkoBeamMaterial(name="TimoshenkoBeam",
                                      model=model, 
                                      elastic_modulus=2.07e11,
                                      poisson_ratio=0.276)
mesh = model.init_mesh()
    
sspace = LagrangeFESpace(mesh=mesh, p=1, ctype='C')
tspace = TensorFunctionSpace(scalar_space=sspace, shape=(6, -1))
integrator = TimoshenkoBeamIntegrator(space=tspace, material=material)

bform = BilinearForm(tspace)
bform.add_integrator(TimoshenkoBeamIntegrator(space=tspace, material=material))
K = bform.assembly().toarray()

F = model.body_force()
F1 = F.reshape(-1)

test = mesh.boundary_face_flag()

dbc = DirichletBC(space=tspace, gd=model.dirichlet, threshold=model.dirichlet_node_index)
K, F = dbc.apply(A=K, f=F)