import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH

pde = cgraph.create("RasingBubble2D")
mesher = cgraph.create("Box2d")
phispacer = cgraph.create("FunctionSpace")
uspacer = cgraph.create("TensorFunctionSpace")
pspacer = cgraph.create("FunctionSpace")

allencahn_eq = cgraph.create("AllenCahnFEMSimulation")
guns_eq = cgraph.create("GaugeUzawaNSSimulation")
solver = cgraph.create("MMGUACNSFEMSolver")

d = 0.005
domain = [-d,d,-2*d,2*d]
area = 2*d * 4*d
rho0 = 1.0
rho1 = 10.0
mu0 = 0.0011
mu1 = 0.0011
gamma = 0.02
epsilon = 0.02 * d
lam = 0.001
q = 4
options = {
    "domain": domain,
    "d": 0.005,
    "area": area,
    "rho0": rho0,
    "rho1": rho1,
    "mu0": mu0,
    "mu1": mu1,
    "epsilon": epsilon,
}
pde(**options)
mesher(mesh_type = "triangle" ,domain = pde().box ,nx = 32 , ny = 64)
phispacer(mesh = mesher(), p=2)
uspacer(mesh = mesher(), p=2)
pspacer(mesh = mesher(), p=1)

allencahn_eq(epsilon = pde().epsilon,gamma = gamma, 
             init_phase = pde().init_phase,phispace = phispacer(),
             q = q)

guns_eq(rho0 = pde().rho0,rho1 = pde().rho1,
        mu0 = pde().mu0,mu1 = pde().mu1,
        lam = lam,gamma = gamma,
        uspace = uspacer(),pspace = pspacer(),phispace = phispacer(),
        q = q)

output_dir = "./"

solver(domain = pde().box,
       dt = 0.001 , nt = 5000,
       uspace = uspacer(),pspace = pspacer(),phispace = phispacer(),
       update_ac = allencahn_eq().update_ac,
       update_us = guns_eq().update_us,
       update_ps = guns_eq().update_ps,
       update_velocity = guns_eq().update_velocity,
       update_gauge = guns_eq().update_gauge,
       update_pressure = guns_eq().update_pressure,
       init_phase = pde().init_phase,
       init_velocity = pde().init_velocity,
       init_pressure = pde().init_pressure,
       velocity_dirichlet_bc = pde().velocity_dirichlet_bc,
       phase_force = pde().phase_force,
       velocity_force = pde().velocity_force,
       output_dir = output_dir)

WORLD_GRAPH.output(u = solver().u ,ux = solver().ux ,uy = solver().uy, 
                   p = solver().p ,phi = solver().phi)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())