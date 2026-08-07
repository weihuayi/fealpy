from fealpy.backend import backend_manager as bm


def test_exp0013_is_time_dependent_divergence_free_mms():
    bm.set_backend("numpy")
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(13)
    point = bm.array([[0.37, 0.41, 0.53]])

    assert pde.geo_dimension() == 3
    assert bm.to_numpy(pde.velocity(point, 0.0)).shape == (1, 3)
    assert float(bm.max(bm.abs(pde.div_velocity(point, 0.3)))) < 1.0e-13
    assert float(bm.max(bm.abs(pde.source(point, 0.3) - pde.source(point, 0.0)))) > 1.0e-8


def test_piso_model_accepts_3d_unsteady_mms_and_nz():
    bm.set_backend("numpy")
    from fealpy.fvm import (
        CollocatedPressureSystemControls,
        NSFVMPISOModel,
        PressureClosureKind,
    )

    model = NSFVMPISOModel(
        {
            "pde": 13,
            "mesh_type": "uniform_tet",
            "nx": 2,
            "ny": 2,
            "nz": 3,
            "duration": (0.0, 0.01),
            "time_steps": 1,
            "n_correctors": 2,
            "momentum_nonorthogonal_max_iterations": 50,
            "pressure_nonorthogonal_max_iterations": 50,
            "pressure_system_controls": (
                CollocatedPressureSystemControls(
                    pure_neumann_closure=PressureClosureKind.GAUGE,
                )
            ),
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    assert model.GD == 3
    assert model.NC == 72

    result = model.solve()
    errors = model.compute_error(result)

    assert result.velocity.shape == (model.NC, model.GD)
    assert result.pressure.shape == (model.NC,)
    assert result.face_velocity.shape == (
        model.mesh.number_of_faces(),
        model.GD,
    )
    assert result.face_flux.shape == (model.mesh.number_of_faces(),)
    assert len(errors) == 4
    assert all(float(error) < 10.0 for error in errors)
