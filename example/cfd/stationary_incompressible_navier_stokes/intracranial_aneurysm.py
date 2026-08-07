from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.model import CFDPDEModelManager

options = {
    'backend': 'numpy',
    'method': 'Ossen',
    'solve': 'direct',
    'apply_bc': 'cylinder',
    'postprocess': 'res',
    'run': 'main',
    'maxit': 1,
    'maxstep': 1000,
    'tol': 1e-12
}

bm.set_backend(options['backend'])
from fealpy.cfd.model.stationary_incompressible_navier_stokes.intracranial_aneurysm_3d import IntracranialAneurysm3d
pde = IntracranialAneurysm3d()
model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh = pde.mesh, options = options)

def to_vtk(uh1, ph1, i):
    mesh.nodedata['ph'] = ph1
    mesh.nodedata['uh'] = uh1.reshape(3,-1).T
    mesh.to_vtk(f'stationary_2d_{i+1}.vtu')

mesh = pde.mesh
maxit = options['maxit']
maxstep = options['maxstep']
tol = options['tol']
uh0 = model.fem.uspace.function()
ph0 = model.fem.pspace.function()
print("ugdof: ", model.fem.uspace.number_of_global_dofs())
print("pgdof: ", model.fem.pspace.number_of_global_dofs())

for j in range(maxstep):

    model.logger.info(f"Iteration {j+1}")
    model.logger.info(f"正在组装算子...")
    BForm, LForm = model.linear_system() 
    model.logger.info(f"算子组装完成")
    model.fem.update(uh0)
    model.logger.info(f"线性系统更新完成")
    model.logger.info(f"正在组装左端项...")
    A = BForm.assembly() 
    model.logger.info(f"左端项组装完成")
    model.logger.info(f"正在组装右端项...")
    b = LForm.assembly()
    model.logger.info(f"右端项组装完成")
    model.logger.info(f"正在处理边界条件...")
    A, b = model.fem.apply_bc(A, b, pde)
    model.logger.info(f"边界处理完成")
    model.logger.info(f"正在求解线性系统...")
    x = model.solve(A, b)
    model.logger.info(f"线性系统求解完成")

    ugdof = model.fem.uspace.number_of_global_dofs()
    uh1= model.fem.uspace.function()
    ph1 = model.fem.pspace.function()
    uh1[:] = x[:ugdof]
    ph1[:] = x[ugdof:]

    to_vtk(uh1, ph1, j)

    res_u = mesh.error(uh0, uh1)
    res_p = mesh.error(ph0, ph1)
    print(f"res_u: {res_u}, res_p: {res_p}")
    if res_u + res_p < tol:
        print(f"Converged at iteration {j+1}")
        break 
    uh0[:] = uh1
    ph0[:] = ph1
