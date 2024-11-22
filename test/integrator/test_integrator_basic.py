
import pytest

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    ScalarRobinSourceIntegrator,
    ScalarDiffusionIntegrator
)

int_types = [
    ScalarRobinSourceIntegrator,
    ScalarDiffusionIntegrator
]


@pytest.mark.parametrize("int_type", int_types)
def test_keep_data(int_type):
    mesh = TriangleMesh.from_box()
    space = LagrangeFESpace(mesh, p=1)
    integrator = int_type(1.)
    integrator.keep_data(True)
    assert len(integrator._cache) == 0
    integrator(space)
    assert len(integrator._cache) > 0
    integrator.keep_data(False)
    assert len(integrator._cache) == 0
    integrator(space)
    assert len(integrator._cache) == 0


@pytest.mark.parametrize("int_type", int_types)
def test_keep_result(int_type):
    mesh = TriangleMesh.from_box()
    space = LagrangeFESpace(mesh, p=1)
    integrator = int_type(1.)
    integrator.keep_result(True)
    assert integrator._value is None
    integrator(space)
    assert integrator._value is not None
    integrator.keep_result(False)
    assert integrator._value is None
    integrator(space)
    assert integrator._value is None
