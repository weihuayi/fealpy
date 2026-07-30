import pytest

from fealpy.fvm import (
    CollocatedPressureSystemControls,
    NSFVMPISOModel,
    NSFVMSimpleModel,
    PoissonFVMModel,
    PressureClosureKind,
    StokesFVMSimpleModel,
)


@pytest.mark.parametrize(
    ("model_class", "pde"),
    [
        (NSFVMSimpleModel, 1),
        (StokesFVMSimpleModel, 1),
        (NSFVMPISOModel, 3),
        (PoissonFVMModel, 2),
    ],
)
def test_fvm_model_applies_every_requested_mesh_refinement(
    model_class,
    pde,
):
    options = {
        "pde": pde,
        "mesh_type": "uniform_tri",
        "nx": 1,
        "ny": 1,
        "mesh_refine": 2,
        "pbar_log": False,
        "log_level": "WARNING",
    }
    if model_class is NSFVMPISOModel:
        options["pressure_system_controls"] = (
            CollocatedPressureSystemControls(
                pure_neumann_closure=PressureClosureKind.GAUGE,
            )
        )

    model = model_class(options)

    assert model.mesh.number_of_cells() == 32
    close = getattr(model, "close", None)
    if close is not None:
        close()
