
from fealpy.pinn.hyperparams import AutoTest


class TestAutotest():
    def test_one(self):
        sets = {
            "a": [1, 2, 3],
            "b": [0.1, 0.01, 0.001],
            "c": [3.1415926, ],
            "d": [None, None]
        }
        at = AutoTest(sets)
        out = list(at.run())

        assert len(out) == 18
        assert len(out[0]) == 4

    def test_two(self):
        sets = {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8],
        }
        at = AutoTest(sets)
        out = list(at.run())

        assert len(out) == 18
        assert len(out[0]) == 3
        assert out[0] == (1, 4, 7)
        assert out[5] == (3, 5, 7)
        assert out[17] == (3, 6, 8)

    def test_three(self):
        sets = {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8],
        }
        at = AutoTest(sets)
        at.set_autosave('example_path', mode='append')
        assert at._autosave is not None
        _ = list(at.run())
        assert at._autosave is not None
        at.set_autosave(None, mode='append')
        assert at._autosave is None
