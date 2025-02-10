from fenics import *
from dolfin_adjoint import *
import cyipopt as ipopt

E, nu = 1e5, 0.3 # Structure material properties
L, H = 3.0, 1.0  # Geometry of the design domain
F = 2000         # Load (T)
p, eps = Constant(3.0), Constant(1.0e-3)      # penalisation and SIMP constants
rho_0, Vol = Constant(0.5), Constant(0.5*L*H) # Top.Opt.constants: Initial guess and Volume constraint

# Mesh constant
nx, ny = 300, 100
mesh = RectangleMesh(MPI.comm_world, Point(0, 0), Point(L, H), nx, ny)

# Define function space and base functions
V, W = VectorFunctionSpace(mesh, "CG", 2), FunctionSpace(mesh, "CG", 1)

# Boundary Condition
left = CompiledSubDomain("near(x[0], 0.0, tol) && on_boundary", tol=1e-14)
bc = [DirichletBC(V, Constant((0.0, 0.0)), left)]

# Load
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
CompiledSubDomain("x[0] == l && x[1] >= (h-0.1)/2 && x[1] <= (h+0.1)/2", l=L, h=H).mark(boundaries, 1)
ds = Measure("ds")(subdomain_data=boundaries)
t = Constant((0.0, -F))

# Forward Function
def forward(rho):
    u, v = TrialFunction(V), TestFunction(V)
    E_rho = eps + (E - eps) * (rho ** p)
    lmbda = nu * E_rho / ((1 + nu) * (1 - 2 * nu))
    mu = E_rho / (2 * (1 + nu))
    a = 2 * mu * inner(sym(grad(u)), sym(grad(v))) * dx + lmbda * div(u) * div(v) * dx
    L = inner(t, v) * ds(1)
    u = Function(V, name="Displacement")
    solve(a == L, u, bc, annotate=True)
    u_values = u.vector().get_local()
    print(f"u_values = {u_values}")
    return u

# Main Code
if __name__ == "__main__":
    rho = interpolate(Constant(float(Vol)), W)
    rho_values = rho.vector().get_local()
    print(f"rho_values = {rho_values}")
    u = forward(rho)
    J = assemble(inner(t, u) * ds(1))
    m = Control(rho)
    Jhat = ReducedFunctional(J, m)
    volume_constraint = UFLInequalityConstraint((Vol - rho) * dx, m) # Volume Constraint
    lb, ub = 0.0, 1.0
    problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    parameters = {"acceptable_tol": 1e-3, "maximum_iterations": 100}
    solver = IPOPTSolver(problem, parameters=parameters)
    rho_opt = solver.solve()

    # Paraview Results
    file_results = XDMFFile("/home/heliang/FEALPy_Development/fealpy/app/soptx/soptx/vtu/solution_cantielver.xdmf")
    file_results.parameters["flush_output"] = True
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(rho_opt, 0)
    file_results.write(u, 0)