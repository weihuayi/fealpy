import pytest
from fealpy.backend.base import Backend

@pytest.fixture
def backend():
    return Backend()

def test_occurrence_empty_iterable(backend):
    items, firsts, lasts = backend.occurrence([])
    assert items == []
    assert firsts == []
    assert lasts == []

def test_occurrence_single_element(backend):
    items, firsts, lasts = backend.occurrence([1])
    assert items == [1]
    assert firsts == [0]
    assert lasts == [0]

def test_occurrence_multiple_elements(backend):
    items, firsts, lasts = backend.occurrence([1, 2, 3, 2, 1])
    assert items == [1, 2, 3]
    assert firsts == [0, 1, 2]
    assert lasts == [4, 3, 2]

def test_occurrence_unordered_elements(backend):
    items, firsts, lasts = backend.occurrence([3, 1, 2, 1, 3])
    assert items == [3, 1, 2]
    assert firsts == [0, 1, 2]
    assert lasts == [4, 3, 2]

def test_occurrence_repeated_elements(backend):
    items, firsts, lasts = backend.occurrence([1, 1, 1, 2, 2])
    assert items == [1, 2]
    assert firsts == [0, 3]
    assert lasts == [2, 4]

def test_occurrence_non_hashable_items(backend):
    with pytest.raises(TypeError):
        backend.occurrence([[1], [2], [1]])
