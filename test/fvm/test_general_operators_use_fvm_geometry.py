import ast
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh import QuadrangleMesh


class ShiftedOperatorGeometry:
    def __init__(self, mesh, *, index=slice(None)):
        from fealpy.fvm.fvm_geometry import FVMGeometry

        real = FVMGeometry(mesh, index=index)
        self._real = real
        is_boundary = np.asarray(real.is_boundary)

        face_center = np.asarray(real.face_center).copy()
        face_center[is_boundary, 0] += 3.0
        face_center[is_boundary, 1] -= 2.0

        S_f = np.asarray(real.S_f).copy()
        S_f *= 2.5

        self.mesh = mesh
        self.index = index
        self.face_to_cell = real.face_to_cell
        self.owner = real.owner
        self.neighbour = real.neighbour
        self.is_internal = real.is_internal
        self.is_boundary = real.is_boundary
        self.cell_center = real.cell_center
        self.face_center = bm.array(face_center)
        self.d_f = real.d_f
        self.mag_d_f = real.mag_d_f
        self.S_f = bm.array(S_f)
        self.mag_S_f = bm.linalg.norm(self.S_f, axis=1)
        self.n_f = self.S_f / self.mag_S_f[:, None]
        self.boundary_owner_to_face_vector = (
            self.face_center[self.is_boundary] - self.cell_center[self.owner[self.is_boundary]]
        )
        self.boundary_normal_distance = bm.einsum(
            "ij,ij->i",
            self.boundary_owner_to_face_vector,
            self.n_f[self.is_boundary],
        )

    def __getattr__(self, name):
        return getattr(self._real, name)

    def linear_owner_weight(self):
        return bm.where(
            self.is_internal,
            0.25 * bm.ones_like(self.mag_S_f),
            1.0,
        )

    def diffusion_face_decomposition(self, method="over_relaxed", *, eps=0.05):
        from fealpy.fvm.fvm_geometry import DiffusionFaceDecomposition

        return DiffusionFaceDecomposition(
            E_f=self.S_f,
            mag_E_f=self.mag_S_f,
            T_f=0.4 * self.S_f,
            orthogonal_factor=self.mag_S_f / self.mag_d_f,
        )

    def scatter_face_flux_to_cells(self, face_flux):
        from fealpy.fvm.fvm_geometry import FVMGeometry

        return FVMGeometry(self.mesh, index=self.index).scatter_face_flux_to_cells(face_flux)


def _quad_mesh():
    bm.set_backend("numpy")
    return QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)


def test_gradient_reconstruct_boundary_data_uses_fvm_geometry(monkeypatch):
    import fealpy.fvm.gradient_reconstruct as gradient_module
    from fealpy.fvm import GradientReconstruct, ResolvedGradientBoundary

    mesh = _quad_mesh()
    monkeypatch.setattr(gradient_module, "FVMGeometry", ShiftedOperatorGeometry)
    geometry = ShiftedOperatorGeometry(mesh)
    field = bm.zeros(mesh.number_of_cells())
    gd = lambda p: p[:, 0] - 0.5 * p[:, 1]
    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    empty_faces = boundary_faces[:0]
    boundary = ResolvedGradientBoundary(
        dirichlet_faces=boundary_faces,
        dirichlet_values=gd(geometry.face_center[boundary_faces]),
        neumann_faces=empty_faces,
        neumann_sn_grad=bm.zeros(0, dtype=field.dtype),
    )
    grad = GradientReconstruct(
        geometry,
        boundary,
    ).cell_gradient(field)

    owner = np.asarray(geometry.owner)[boundary_faces]
    nonzero_owner = set(np.where(np.linalg.norm(np.asarray(grad), axis=1) > 1.0e-12)[0])

    assert nonzero_owner == set(owner.tolist())


def test_convection_integrator_uses_fvm_geometry_face_area_vector(monkeypatch):
    import fealpy.fvm.convection_integrator as convection_module
    from fealpy.fvm import ConvectionIntegrator, face_interpolation_owner_weight

    mesh = _quad_mesh()
    monkeypatch.setattr(convection_module, "FVMGeometry", ShiftedOperatorGeometry)
    space = ScaledMonomialSpace2d(mesh, 0)

    local = ConvectionIntegrator(
        coef=bm.ones((mesh.number_of_faces(), mesh.geo_dimension())),
        interpolation="linear",
        q=1,
    ).assembly(space)

    geometry = ShiftedOperatorGeometry(mesh)
    flux = np.einsum(
        "ij,ij->i",
        np.ones((mesh.number_of_faces(), mesh.geo_dimension())),
        np.asarray(geometry.S_f),
    )
    owner_weight = np.asarray(
        face_interpolation_owner_weight(geometry, method="linear")
    )
    expected = flux[:, None, None] * np.array(
        [
            [[weight, 1.0 - weight], [-weight, weight - 1.0]]
            for weight in owner_weight
        ]
    )

    np.testing.assert_allclose(np.asarray(local), expected, rtol=1.0e-13, atol=1.0e-13)


def test_collocated_divergence_and_mass_residual_use_fvm_geometry():
    from fealpy.fvm import collocated_mass_residual

    mesh = _quad_mesh()
    face_velocity = bm.ones((mesh.number_of_faces(), mesh.geo_dimension()))

    geometry = ShiftedOperatorGeometry(mesh)
    face_flux = np.einsum("ij,ij->i", np.ones_like(np.asarray(geometry.S_f)), np.asarray(geometry.S_f))
    div = geometry.scatter_face_flux_to_cells(face_flux)
    residual = collocated_mass_residual(
        face_velocity,
        geometry=geometry,
    )
    expected_div = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    np.testing.assert_allclose(np.asarray(div), expected_div, rtol=1.0e-13, atol=1.0e-13)
    assert residual >= 0.0


def test_rhie_chow_consumes_the_solver_pressure_gradient():
    from fealpy.fvm import NSFVMSimpleModel

    model = NSFVMSimpleModel({
        "pde": 6,
        "nx": 2,
        "ny": 2,
        "pbar_log": False,
        "log_level": "ERROR",
    })
    rhie_chow = model.solver.rhie_chow

    assert not hasattr(rhie_chow, "gradient_reconstruct")
    assert (
        model.solver.pressure_gradient.geometry
        is rhie_chow.geometry
    )


def test_core_geometry_work_arrays_declare_their_device():
    """Backend-created arrays must follow an existing tensor explicitly."""
    repository = Path(__file__).resolve().parents[2]
    sources = (
        repository / "fealpy/fvm/fvm_geometry.py",
        repository / "fealpy/fvm/gradient_reconstruct.py",
    )

    missing = []
    for source in sources:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "bm"
                and function.attr in {"array", "arange", "zeros"}
            ):
                continue
            if any(keyword.arg == "device" for keyword in node.keywords):
                continue
            missing.append(f"{source.name}:{node.lineno}:{function.attr}")

    assert missing == []
