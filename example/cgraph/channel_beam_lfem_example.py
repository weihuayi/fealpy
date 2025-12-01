import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH
model = cgraph.create("ChannelBeam3d")
mesher = cgraph.create("ChannelBeamMesh")
spacer = cgraph.create("FunctionSpace")
materialer = cgraph.create("ChannelBeamMaterial")
ChannelBeam_model = cgraph.create("ChannelBeam")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
coord = cgraph.create("Rbeam3d")
strain_stress = cgraph.create("ChannelStrainStress")

# 连接节点
spacer(type="lagrange", mesh=mesher(), p=1)
materialer(
    property="structural-steel",
    type="Timoshenko",
    mu_y=model().mu_y,
    mu_z=model().mu_z,
    E=2.1e11, nu=0.3, density=7800)

ChannelBeam_model(
    mu_y = model().mu_y,
    mu_z = model().mu_z,
    GD = model().GD,
    space = spacer(),
    beam_E = materialer().E,
    beam_nu = materialer().nu,
    beam_density = materialer().rho,
    load_case = model().load_case,
    gravity = 9.81,
    dirichlet_dof = model().dirichlet_dof
)

solver(A = ChannelBeam_model().K,
       b = ChannelBeam_model().F)

postprocess(out = solver().out, node_ldof=6, type="Timo_beam")

coord(mesh=mesher(), vref=None, index=None)

strain_stress(
    mu_y = model().mu_y,
    mu_z = model().mu_z,
    E = materialer().E,
    nu = materialer().nu,
    mesh=mesher(),
    uh = solver().out,
    coord_transform=coord().R,
    y = 0.0,
    z = 0.0
)


# 最终连接到图输出节点上
# WORLD_GRAPH.output(material=materialer())
WORLD_GRAPH.output(out=solver().out,
                   strain=strain_stress().strain, stress=strain_stress().stress
                   )
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())