
import torch
from torch import Tensor, sin, tensordot


BATCH_SIZE = 10
MAXIT = 3

def gd(p: Tensor):
    x = p[..., 0]
    y = p[..., 1]
    theta = torch.atan2(y, x)
    omega = torch.arange(1, BATCH_SIZE+1, dtype=p.dtype, device=p.device)
    return sin(tensordot(omega, theta, dims=0))


def test_1():
    from lafemeit.solver import LaplaceFEMSolver

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)
    perr = []
    uerr = []

    for i in range(MAXIT):
        solver = LaplaceFEMSolver(mesh, reserve_matrix=True)
        ipoint = mesh.interpolation_points(p=1)[solver.space.is_boundary_dof()]
        g = gd(ipoint)
        g -= torch.mean(g, dim=-1, keepdim=True)

        phi = solver.solve_from_potential(g)
        current = solver.normal_derivative(phi)
        phi2 = solver.solve_from_current(current)
        potential = solver.boundary_value(phi2)

        uerr.append(torch.sum(torch.abs(phi - phi2)) / phi.numel())
        perr.append(torch.sum(torch.abs(potential - g)) / potential.numel())

        mesh.uniform_refine()

    print(uerr, perr, sep='\n')


def test_2():
    from lafemeit.solver import LaplaceFEMSolver

    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10)
    perr = []
    uerr = []

    for _ in range(MAXIT):
        solver = LaplaceFEMSolver(mesh, reserve_matrix=True)
        ipoint = mesh.interpolation_points(p=1)[solver.space.is_boundary_dof()]
        g = gd(ipoint)
        g -= torch.mean(g, dim=-1, keepdim=True)

        phi = solver.solve_from_current(g)
        potential = solver.boundary_value(phi)
        phi2 = solver.solve_from_potential(potential)
        current = solver.normal_derivative(phi2)

        uerr.append(torch.sum(torch.abs(phi - phi2)) / phi.numel())
        perr.append(torch.sum(torch.abs(current - g)) / potential.numel())

        mesh.uniform_refine()

    print(uerr, perr, sep='\n')


if __name__ == '__main__':
    from fealpy.backend import backend_manager as bm
    from fealpy.mesh import TriangleMesh

    bm.set_backend('pytorch')

    test_1()
    test_2()
