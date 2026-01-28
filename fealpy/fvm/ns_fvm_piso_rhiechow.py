from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager
from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import ScaledMonomialSpace2d,TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.solver import spsolve

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ConvectionIntegrator,
    ScalarSourceIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
    RhieChowInterpolation
)
def sum_duplicates_csr_manual(csr):
    from fealpy.sparse import csr_matrix
    indptr = csr.indptr       # shape (nrow+1,)
    indices = csr.indices     # shape (nnz,)
    data = csr.data           # shape (nnz,)
    nrow, ncol = csr.shape
    counts = indptr[1:] - indptr[:-1]
    row = bm.repeat(bm.arange(nrow), counts)
    flat_idx = row * ncol + indices
    summed = bm.bincount(flat_idx, weights=data, minlength=nrow*ncol)
    nnz_idx = bm.nonzero(summed)[0]
    new_data = summed[nnz_idx]
    new_row, new_col = divmod(nnz_idx, ncol)
    return csr_matrix((new_data, (new_row, new_col)), shape=csr.shape)
from fealpy.decorator import cartesian

duration = [0, 1]
nt = 40
tau = (duration[1] - duration[0]) / nt

pde = PDEModelManager("navier_stokes").get_example(3)

nx = ny = 40
mesh = pde.init_mesh['uniform_qrad'](nx=nx, ny=ny)

space = ScaledMonomialSpace2d(mesh, p=0)
velocity_space = TensorFunctionSpace(space, shape=(2, -1))
cm = mesh.entity_measure("cell")
points = mesh.entity_barycenter("cell")
epoints = mesh.entity_barycenter("edge")
NC = mesh.number_of_cells()


U0 = pde.velocity_0(points, 0)
Uf0 = pde.velocity_0(epoints, 0)
p0 = pde.pressure_0(points, 0)

def temporary_velocity(U0,Uf0, p0, t):

    bform = BilinearForm(velocity_space)
    bform.add_integrator(ScalarDiffusionIntegrator(q=2))
    bform.add_integrator(ConvectionIntegrator(q=2, coef=Uf0))
    A = bform.assembly()

    M = CSRTensor(
        crow=bm.arange(2*NC+1),
        col=bm.arange(2*NC),
        values=bm.concatenate([cm, cm]),
        spshape=(2*NC, 2*NC)
    )

    @cartesian
    def src(p):
        return pde.source(p, t)

    f = LinearForm(velocity_space).add_integrator(
        ScalarSourceIntegrator(src, q=2)).assembly()

    grad_p = GradientReconstruct(mesh).LSQ(p0)
    p1 = bm.einsum('i,i->i', grad_p[:,0], cm)
    p2 = bm.einsum('i,i->i', grad_p[:,1], cm)
    p_grad_integrator = bm.concatenate((p1,p2))
    A = tau*A + M
    b = tau*(f - p_grad_integrator) + (U0*cm[:, None]).flatten(order='F')
    dbc = DirichletBC(mesh, pde.velocity_dirichlet)
    A, b = dbc.DiffusionApply(A, b)
    A = sum_duplicates_csr_manual(A)
    a_p = A.diags().values
    U = spsolve(A, b, "mumps")
    return U, a_p

def solve_pressure_correction(rhs, a_p):
    dp = cm/a_p[:len(cm)]
    e2c = mesh.edge_to_cell()
    coef = (dp[e2c[:,0]]+dp[e2c[:,1]])/2

    A = BilinearForm(space).add_integrator(
        ScalarDiffusionIntegrator(q=2, coef=coef)
    ).assembly()

    A1 = COOTensor(
        bm.array([bm.zeros(len(cm), dtype=bm.int32),
                  bm.arange(len(cm), dtype=bm.int32)]),
        cm,
        spshape=(1, len(cm))
    )
    A = BlockForm([[A, A1.T], [A1, None]])
    A = A.assembly_sparse_matrix(format='csr')
    b = bm.concatenate([rhs, bm.array([0])], axis=0)
    sol = spsolve(A, b, "scipy")
    return sol[:-1]



def solve(U0,Uf0,p0):
    for n in range(nt):
        print(n)
        t = duration[0] + n * tau
        u_tem, a_p = temporary_velocity(U0,Uf0, p0, t)
        uf = RhieChowInterpolation(mesh).Interpolation(u_tem,a_p,p0)   
        div_rhs1 = DivergenceReconstruct(mesh).Reconstruct(uf)
        p_corr1 = solve_pressure_correction(-div_rhs1, a_p)
        p1 = p0 + p_corr1
        
        grad_p1 = GradientReconstruct(mesh).LSQ(p_corr1)
        u_tem = bm.stack([u_tem[:NC],u_tem[NC:]],axis=-1)
        u_tem2 = u_tem - (cm/a_p[:NC])[:,None]*grad_p1
        u_tem2 = u_tem2.flatten(order='F')
        uf2 = RhieChowInterpolation(mesh).Interpolation(u_tem2,a_p,p1) 
        div_rhs2 = DivergenceReconstruct(mesh).Reconstruct(uf2)
        p_corr2 = solve_pressure_correction(-div_rhs2, a_p)
        p2 = p1 + p_corr2

        grad_p2 = GradientReconstruct(mesh).LSQ(p_corr2)
        u_tem2 = bm.stack([u_tem2[:NC],u_tem2[NC:]],axis=-1)
        u_tem3 = u_tem2 - (cm/a_p[:NC])[:,None]*grad_p2
        u_tem3 = u_tem3.flatten(order='F')
        U0 = bm.stack([u_tem3[:NC],u_tem3[NC:]],axis=-1)
        Uf0 = RhieChowInterpolation(mesh).Interpolation(u_tem3,a_p,p2)
        p0 = p2

    return u_tem3[:NC],u_tem3[NC:],p2



uh, vh, ph = solve(U0,Uf0,p0)

uI = pde.velocity_u(points, 1)
vI = pde.velocity_v(points, 1)
pI = pde.pressure(points, 1)

L2u = bm.sqrt(bm.sum(cm * (uh - uI) ** 2))
L2v = bm.sqrt(bm.sum(cm * (vh - vI) ** 2))
L2p = bm.sqrt(bm.sum(cm * (ph - pI) ** 2))

print("Collocated PISO:")
print("L2 error of u:", L2u)
print("L2 error of v:", L2v)
print("L2 error of p:", L2p)


import matplotlib.pyplot as plt
cell_centers = mesh.entity_barycenter('cell')
x, y = cell_centers[:, 0], cell_centers[:, 1]

fig = plt.figure(figsize=(15, 10))
titles = [
    ("Error u", uh - uI),
    ("Error v", vh - vI),
    ("Error p", ph - pI),
]
for i, (title, data) in enumerate(titles):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    ax.plot_trisurf(x, y, data, cmap='viridis')
    ax.set_title(title)
plt.tight_layout()
plt.show()