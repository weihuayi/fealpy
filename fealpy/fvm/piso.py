from typing import Union, Tuple
from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor,CSRTensor
from fealpy.functionspace import ScaledMonomialSpace2d,TensorFunctionSpace  
from fealpy.fem import BilinearForm, LinearForm, BlockForm
from fealpy.solver import spsolve
from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ConvectionIntegrator,
    ScalarSourceIntegrator,
    StaggeredMeshManager,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
    VectorDecomposition,
    NeumannBC,
    RhieChowInterpolation)



duration = [0,1]
nt = 20
tau = (duration[1] -duration[0]) / nt


pde = PDEModelManager("navier_stokes").get_example(1)
nx = 3
ny = 3
staggered_mesh = StaggeredMeshManager(pde.domain(), nx=nx, ny=ny)
umesh = staggered_mesh.umesh
vmesh = staggered_mesh.vmesh
pmesh = staggered_mesh.pmesh
ucell2pedge = staggered_mesh.get_dof_mapping_ucell2pedge()

ppoints = pmesh.entity_barycenter("cell")
p = pde.pressure(ppoints)
# e, d = VectorDecomposition(pmesh).centroid_vector_calculation()
# partial_p = (p[pmesh.edge_to_cell()[:,1]] - p[pmesh.edge_to_cell()[:,0]])/d
# e_cf = e / d[:, None]
# # print(e_cf)
# grad_p = GradientReconstruct(pmesh).AverageGradientreNeumann(
#             p, pde.neumann_pressure)
grad_p = GradientReconstruct(pmesh).test(p)
grad_p_I = pde.grad_pressure(ppoints)
pcm = pmesh.entity_measure("cell")
L2error = bm.sqrt(bm.sum(pcm * (grad_p[:,0] - grad_p_I[:,0]) ** 2))
print("error of grad p at cell:",L2error)
# print("error of grad p at face:",grad_p[3,0] - grad_p_I[3,0])
exit()
# overline_grad_p_f = GradientReconstruct(pmesh).reconstruct(grad_p)
# GradientDifference = (partial_p - bm.einsum('ij,ij->i', overline_grad_p_f, e_cf))[:, None]*e_cf
# # print("GradientDifference:",GradientDifference)
# gradp_f = overline_grad_p_f + GradientDifference
# pem = pmesh.entity_measure("edge")
# pepoints = pmesh.entity_barycenter("edge")
# grad_p_f_I = pde.grad_pressure(pepoints)
# error = grad_p_f_I - gradp_f
# # L2error = bm.sqrt(bm.sum(pem * (GradientDifference[:,0]) ** 2))
# L2error1 = bm.sqrt(bm.sum(pem * (grad_p_f_I[:,0] - overline_grad_p_f[:,0]) ** 2))
# L2error2 = bm.sqrt(bm.sum(pem * (grad_p_f_I[:,0] - gradp_f[:,0]) ** 2))
# error = bm.max(bm.abs(GradientDifference[:,0]))
# # error = bm.max(bm.abs(grad_p_f_I[:,0] - gradp_f[:,0]))
# print("error of grad p at face:",L2error1)
# print("error of grad p at face:",L2error2)
# exit()

from matplotlib import pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111)
pmesh.add_plot(axes) # 画出网格背景
pmesh.find_edge(axes, showindex=True) # 找到单元重心
plt.show()
exit()



rhie_chow = RhieChowInterpolation(pmesh)
space = ScaledMonomialSpace2d(pmesh,p=0) 
bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(q=2))
A = bform.assembly()
uap = A.diags().values
cm = pmesh.entity_measure("cell")
em = pmesh.entity_measure("edge")
dp = cm/uap
# print("a_p:",dp)
cellpoints = pmesh.entity_barycenter("cell")
facepoints = pmesh.entity_barycenter("face")
uf = pde.velocity(facepoints)
u = pde.velocity(cellpoints)
uf_rhie,df = rhie_chow.Ucell2edge(u,dp)
u_difference = uf - uf_rhie 
print(u_difference[0,0])
p = pde.pressure(cellpoints)
p_defference = rhie_chow.GradientDifference(p)
print(p_defference[80,:])
exit()

# print(pmesh.cell_to_edge())
# print("ucell2pedge:",ucell2pedge)

# pe2c = pmesh.edge_to_cell()
# print("pe2c:",pe2c[ucell2pedge,0])
# exit()
uspace = ScaledMonomialSpace2d(umesh,p=0)
vspace = ScaledMonomialSpace2d(vmesh,p=0)
pspace = ScaledMonomialSpace2d(pmesh,p=0)
pcm = pmesh.entity_measure("cell")
ucm = umesh.entity_measure("cell")
vcm = vmesh.entity_measure("cell")
ppoints = pmesh.entity_barycenter("cell")
upoints = umesh.entity_barycenter("cell")
vpoints = vmesh.entity_barycenter("cell")
UNC = umesh.number_of_cells()
VNC = vmesh.number_of_cells()
PNC = pmesh.number_of_cells()
u0 = pde.velocity_u0(upoints,0)
v0 = pde.velocity_v0(vpoints,0)
p0 = pde.pressure_0(ppoints,0)
from fealpy.decorator import cartesian


def compute_temporary_velocity_u(u0,v0,p_u,t):
    ue2c = umesh.edge_to_cell()
    ve2c = vmesh.edge_to_cell()
    uf0 = (u0[ue2c[:,0]] + u0[ue2c[:,1]])/2
    vf0 = (v0[ve2c[:,0]] + v0[ve2c[:,1]])/2
    vedge2uedge = staggered_mesh.get_dof_mapping_vedge2uedge()
    vf0_umesh = vf0[vedge2uedge.astype(int)]
    Uf0 = bm.stack([uf0, vf0_umesh], axis=1)
    bform = BilinearForm(uspace)
    bform.add_integrator(ScalarDiffusionIntegrator(q=2))
    bform.add_integrator(ConvectionIntegrator(q=2, coef=Uf0))
    A = bform.assembly()
    M1 = CSRTensor(crow = bm.arange(UNC+1),
        col = bm.arange(UNC),
        values=ucm,
        spshape=(UNC, UNC))
    @cartesian
    def coef(p):
        time = t
        val = pde.source_u(p, time)
        return val
    fu = LinearForm(uspace).add_integrator(
            ScalarSourceIntegrator(coef, q=2)).assembly()
    grad_p = GradientReconstruct(umesh).test(p_u)
    press = bm.einsum('i,i->i', grad_p[:, 0], ucm)
    mass = bm.einsum('i,i->i', u0, ucm)
    A = A*tau + M1 
    fu = tau*(fu - press) + mass
    dbc = DirichletBC(umesh, pde.velocity_dirichlet_u,
        threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
    A, fu = dbc.DiffusionApply(A, fu)
    A, fu = dbc.ThresholdApply(A, fu)
    uap = A.diags().values
    return spsolve(A, fu,"mumps"), uap



def compute_temporary_velocity_v(u0,v0,p_v,t):
    ue2c = umesh.edge_to_cell()
    ve2c = vmesh.edge_to_cell()
    uf0 = (u0[ue2c[:,0]] + u0[ue2c[:,1]])/2
    vf0 = (v0[ve2c[:,0]] + v0[ve2c[:,1]])/2
    uedge2vedge = staggered_mesh.get_dof_mapping_uedge2vedge()
    uf0_vmesh = uf0[uedge2vedge.astype(int)]
    Uf0 = bm.stack([uf0_vmesh, vf0], axis=1)
    bform = BilinearForm(vspace)
    bform.add_integrator(ScalarDiffusionIntegrator(q=2))
    bform.add_integrator(ConvectionIntegrator(q=2, coef=Uf0))
    A = bform.assembly()
    M2 = CSRTensor(crow = bm.arange(VNC+1),
        col = bm.arange(VNC),
        values=vcm,
        spshape=(VNC, VNC))
    @cartesian
    def coef(p):
        time = t
        val = pde.source_v(p, time)
        return val
    fv = LinearForm(vspace).add_integrator(
            ScalarSourceIntegrator(coef, q=2)).assembly()
    grad_p = GradientReconstruct(vmesh).test(p_v)
    press = bm.einsum('i,i->i', grad_p[:, 1], vcm)
    mass = bm.einsum('i,i->i', v0, vcm)
    A = tau*A + M2 
    fv = tau*(fv - press) + mass
    dbc = DirichletBC(vmesh, pde.velocity_dirichlet_v,
        threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10))
    A, fv = dbc.DiffusionApply(A, fv)
    A, fv = dbc.ThresholdApply(A, fv)
    vap = A.diags().values
    return spsolve(A, fv,"mumps"), vap


def correct_pressure_compute(f: TensorLike, a_p_edge: TensorLike) -> TensorLike:
    pspace = ScaledMonomialSpace2d(pmesh, 0)
    p_edge = pmesh.entity_measure('edge')
    p_edge2 = bm.einsum('i,i->i', p_edge,p_edge)
    A = BilinearForm(pspace).add_integrator(
        ScalarDiffusionIntegrator(q=2,coef=p_edge2 / a_p_edge)
    ).assembly()  
    # print(1 / a_p_edge)
    # print(A.to_dense())
    # nbc = NeumannBC(pmesh, pde.neumann_pressure)
    # f = nbc.DiffusionApply(f)
    LagA = pmesh.entity_measure('cell')
    A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                         bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))
    A = BlockForm([[A, A1.T], [A1, None]])
    A = A.assembly_sparse_matrix(format='csr')
    b0 = bm.array([0])
    b = bm.concatenate([f, b0], axis=0)
    sol = spsolve(A, b,"scipy")
    p_correct = sol[:-1]
    return p_correct

def solve(u0,v0,p0):
    for n in range(nt):
        t = duration[0] + n * tau
        p_u, p_v = staggered_mesh.map_pressure_pcell_to_uvedge(p0)
        u1,uap = compute_temporary_velocity_u(u0,v0,p_u,t)
        v1,vap = compute_temporary_velocity_v(u0,v0,p_v,t)
        
        edge_vel1, a_p_edge = staggered_mesh.map_velocity_uvcell_to_pedge(u1, v1, uap, vap)
        div_rhs1 = DivergenceReconstruct(pmesh).StagReconstruct(edge_vel1)
        p_corr1 = correct_pressure_compute(-div_rhs1, a_p_edge)
        p1 = p0 + p_corr1
        u_pcorr1,v_pcorr1 = staggered_mesh.map_pressure_pcell_to_uvedge(p_corr1)
        ugrad_p1 = GradientReconstruct(umesh).test(u_pcorr1)
        vgrad_p1 = GradientReconstruct(vmesh).test(v_pcorr1)
        nabla_px1 = ugrad_p1[:, 0]
        nabla_py1 = vgrad_p1[:, 1]
        u2 = u1 - 1/uap*nabla_px1
        v2 = v1 - 1/vap*nabla_py1
        # u2 = u1 - ucm/uap*nabla_px1
        # v2 = v1 - vcm/vap*nabla_py1


        edge_vel2, _ = staggered_mesh.map_velocity_uvcell_to_pedge(u2, v2, uap, vap)
        div_rhs2 = DivergenceReconstruct(pmesh).StagReconstruct(edge_vel2)
        p_corr2 = correct_pressure_compute(-div_rhs2, a_p_edge)
        p2 = p1 + p_corr2
        u_pcorr2,v_pcorr2 = staggered_mesh.map_pressure_pcell_to_uvedge(p_corr2)
        ugrad_p2 = GradientReconstruct(umesh).test(u_pcorr2)
        vgrad_p2 = GradientReconstruct(vmesh).test(v_pcorr2)
        nabla_px2 = ugrad_p2[:, 0]
        nabla_py2 = vgrad_p2[:, 1]
        u3 = u2 - 1/uap*nabla_px2
        v3 = v2 - 1/vap*nabla_py2
        # u3 = u2 - ucm/uap*nabla_px2
        # v3 = v2 - vcm/vap*nabla_py2

        u0 = u3
        v0 = v3
        p0 = p2

        # edge_vel3, _ = staggered_mesh.map_velocity_uvcell_to_pedge(u3, v3, uap, vap)
        # div_rhs3 = DivergenceReconstruct(pmesh).StagReconstruct(edge_vel3)
        # p_corr3 = correct_pressure_compute(-div_rhs3, a_p_edge)
        # p3 = p2 + p_corr3
        # u_pcorr3,v_pcorr3 = staggered_mesh.map_pressure_pcell_to_uvedge(p_corr3)
        # ugrad_p3 = GradientReconstruct(umesh).test(u_pcorr3)
        # vgrad_p3 = GradientReconstruct(vmesh).test(v_pcorr3)
        # nabla_px3 = ugrad_p3[:, 0]
        # nabla_py3 = vgrad_p3[:, 1]
        # u4 = u3 - tau/uap*nabla_px3
        # v4 = v3 - tau/vap*nabla_py3

        # u0 = u4
        # v0 = v4
        # p0 = p3
    return u3,v3,p2


def solve2(u0,v0):
    for n in range(nt):
        t = duration[0] + n * tau
        pu = pde.pressure(upoints,t)
        pv = pde.pressure(vpoints,t)
        u1,uap = compute_temporary_velocity_u(u0,v0,pu,t)
        v1,vap = compute_temporary_velocity_v(u0,v0,pv,t)
        u0 = u1
        v0 = v1
    return u1,v1
        

# uh,vh,ph = solve(u0,v0,p0)
uh,vh = solve2(u0,v0)

uI = pde.velocity_u(upoints,1)
vI = pde.velocity_v(vpoints,1)
pI = pde.pressure(ppoints,1)
L2u = bm.sqrt(bm.sum(ucm * (uh - uI) ** 2))
L2v = bm.sqrt(bm.sum(vcm * (vh - vI) ** 2))
# L2p = bm.sqrt(bm.sum(pcm * (ph - pI) ** 2))
print(f"L2 error of u: {L2u}")
print(f"L2 error of v: {L2v}")
# print(f"L2 error of p: {L2p}")
exit()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 5))

px, py = pmesh.entity_barycenter("cell").T
ux, uy = umesh.entity_barycenter("cell").T
vx, vy = vmesh.entity_barycenter("cell").T

# ax1 = fig.add_subplot(1, 3, 1, projection="3d")
# ax1.plot_trisurf(px, py, ph-pI, cmap="viridis")
# ax1.set_title("Pressure")

ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_trisurf(ux, uy, uh-uI, cmap="viridis")
ax2.set_title("U velocity") 

ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot_trisurf(vx, vy, vh-vI, cmap="viridis")
ax3.set_title("V velocity")

plt.tight_layout()
plt.show()