import fealpy.cgraph as cgraph
from fealpy.backend import backend_manager as bm

WORLD_GRAPH = cgraph.WORLD_GRAPH

model = cgraph.create("Timoaxle3d")
mesher = cgraph.create("TimobeamAxleMesh")
spacer = cgraph.create("FunctionSpace")
beam_materialer = cgraph.create("TimoMaterial")
axle_materialer = cgraph.create("AxleMaterial")
timoaxle_model = cgraph.create("Timoaxle")
solver = cgraph.create("DirectSolver")
postprocess = cgraph.create("UDecoupling")
indices = cgraph.create("BeamAxleIndices")
coord1 = cgraph.create("Rbeam3d")
coord2 = cgraph.create("Rbeam3d")
strain_stress = cgraph.create("TimoAxleStrainStress")

# 连接节点
spacer(type="lagrange", mesh=mesher(), p=1)
# spacer(type="lagrange", mesh=mesher(), p=1, gd=6)
beam_materialer(
    property="structural-steel", 
    type="Timoshenko", 
    beam_para = model().beam_para,
    axle_para = model().axle_para,
    E=2.1e11, nu=0.3)
axle_materialer(
    property="structural-steel", 
    type="spring", 
    beam_para = model().beam_para,
    axle_para = model().axle_para,
    E=1.976e6, nu=-0.5)

timoaxle_model(
    beam_para = model().beam_para,
    axle_para = model().axle_para,
    GD = model().GD,
    space = spacer(),
    beam_E = beam_materialer().E,
    beam_nu = beam_materialer().nu,
    axle_E = axle_materialer().E,
    axle_nu = axle_materialer().nu,
    NC = mesher().NC,
    external_load = model().external_load,
    dirichlet_dof = model().dirichlet_dof,
    penalty = 1e20
)

solver(A = timoaxle_model().K,
       b = timoaxle_model().F)

postprocess(out = solver().out, node_ldof=6, type="Timo_beam")

indices(
    beam_num = 22,
    axle_num = 10,
    total_num = mesher().NC
)

R1 = coord1(mesh=mesher(), vref=[0, 1, 0], index = indices().beam_indices)
R2 = coord2(mesh=mesher(), vref=[0, 1, 0], index = indices().axle_indices)

strain_stress(
    beam_para = model().beam_para,
    axle_para = model().axle_para,
    beam_E = beam_materialer().E,
    beam_nu = beam_materialer().nu,
    axle_E = axle_materialer().E,
    axle_nu = axle_materialer().nu,
    mesh=mesher(), 
    uh = solver().out,
    y = 0.0,
    z = 0.0,
    axial_position = None,
    R1 = R1,
    R2 = R2,
    beam_indices = indices().beam_indices,
    axle_indices = indices().axle_indices,
    total_num = mesher().NC
)


# 最终连接到图输出节点上
# WORLD_GRAPH.output(material=beam_materialer(), axle_material=axle_materialer())
WORLD_GRAPH.output(out=solver().out,strain=strain_stress().strain, stress=strain_stress().stress)
WORLD_GRAPH.register_error_hook(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())