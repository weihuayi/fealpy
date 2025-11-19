
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("LinearElasticityEigen3d")           
mesher = cgraph.create("Box3d")                      
spacer = cgraph.create("TensorFunctionSpace")        
isDDof = cgraph.create("BoundaryDof")
eig_eq = cgraph.create("LinearElasticityEigenEquation")
eigensolver = cgraph.create("SLEPcEigenSolver")
dbc = cgraph.create("DirichletBC")

spacer(mesh=mesher(), p=1,gd=3)

eig_eq(space=spacer(), q=3, material = pde().material, displacement_bc = pde().displacement_bc, is_displacement_boundary = pde().is_displacement_boundary)

eigensolver(
    S=eig_eq().stiffness,
    M=eig_eq().mass,
    neigen=6,
)

WORLD_GRAPH.output(eig_eq=eigensolver().val, uh=eigensolver().vec)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())