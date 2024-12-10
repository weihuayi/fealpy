
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    ScalarRobinSourceIntegrator,
    ScalarDiffusionIntegrator,
    ScalarMassIntegrator,
    GroupIntegrator
)

int_types = [
    ScalarRobinSourceIntegrator,
    ScalarDiffusionIntegrator
]

backends = ['numpy', 'pytorch']


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


# @pytest.mark.parametrize("int_type", int_types)
# def test_keep_result(int_type):
#     mesh = TriangleMesh.from_box()
#     space = LagrangeFESpace(mesh, p=1)
#     integrator = int_type(1.)
#     integrator.keep_result(True)
#     assert integrator._cached_output is None
#     integrator(space)
#     assert integrator._cached_output is not None
#     integrator.keep_result(False)
#     assert integrator._cached_output is None
#     integrator(space)
#     assert integrator._cached_output is None


@pytest.mark.parametrize('backend', backends)
def test_group_integrator(backend):
    bm.set_backend(backend)
    mesh = TriangleMesh.from_box()
    space = LagrangeFESpace(mesh, p=1)
    int1 = ScalarDiffusionIntegrator(1.)
    int2 = ScalarMassIntegrator(0.5)
    int_1_2 = int1 + int2
    int3 = ScalarRobinSourceIntegrator(1.)
    int_3_1_2 = int3 + int_1_2
    # Test type
    assert isinstance(int_1_2, GroupIntegrator)
    assert isinstance(int_3_1_2, GroupIntegrator)
    assert int_3_1_2._region is None
    # Test the order of subints
    for subint, expected in zip(int_3_1_2, [int3, int1, int2]):
        assert subint is expected
    # Test iadd
    total = int1
    total += int_3_1_2
    assert total is not int_3_1_2
    total = int_3_1_2
    total += int1
    assert total is int_3_1_2 # now int_3_1_2 is actually like "int_3_1_2_1"
    for subint, expected in zip(int_3_1_2, [int3, int1, int2, int1]):
        assert subint is expected

    # Test calculation
    result = int_1_2.assembly(space)
    assert bm.allclose(result, int1.assembly(space) + int2.assembly(space))
