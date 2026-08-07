import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.fvm import (
    DeviatoricStressSourceIntegrator,
    FVMGeometry,
)
from fealpy.mesh import TriangleMesh


def test_deviatoric_stress_integrator_reuses_fvm_geometry_scatter(monkeypatch):
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=1)
    space = ScaledMonomialSpace2d(mesh, 0)
    geometry = FVMGeometry(mesh)
    nf = mesh.number_of_faces()

    grad_f = np.zeros((nf, mesh.geo_dimension(), mesh.geo_dimension()))
    grad_f[:, 0, 0] = np.linspace(0.1, 0.4, nf)
    grad_f[:, 0, 1] = np.linspace(-0.2, 0.3, nf)
    grad_f[:, 1, 0] = np.linspace(0.5, 0.8, nf)
    grad_f[:, 1, 1] = np.linspace(-0.4, 0.2, nf)
    coef = np.linspace(0.8, 1.3, nf)

    div_u = np.einsum("fii->f", grad_f)
    face_flux = coef[:, None] * (
        np.einsum("fji,fj->fi", grad_f, np.asarray(geometry.S_f))
        - (2.0 / 3.0) * div_u[:, None] * np.asarray(geometry.S_f)
    )
    expected = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    calls = []
    original_scatter = FVMGeometry.scatter_face_flux_to_cells

    def counted_scatter(self, selected_face_flux):
        calls.append(np.asarray(selected_face_flux).copy())
        return original_scatter(self, selected_face_flux)

    monkeypatch.setattr(FVMGeometry, "scatter_face_flux_to_cells", counted_scatter)

    rhs = DeviatoricStressSourceIntegrator(grad_f, coef=coef).assembly(space)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], face_flux, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(rhs), expected, rtol=1.0e-13, atol=1.0e-13)
