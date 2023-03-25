
import torch
from fealpy.pinn.tools import mkfs


class TestMkfs():
    def test_one(self):
        a = torch.randn(2, 3)
        b = 1.5
        c = torch.tensor([[0.1], [0.2]])
        result = mkfs(a, b, c)
        assert result.shape == (2, 5)

        result = mkfs(b, c, a)
        assert result.shape == (2, 5)

    def test_two(self):
        a = torch.ones((2, 3), dtype=torch.float32)
        assert mkfs(a).shape == (2, 3)

    def test_three(self):
        a = 1
        b = 1.5
        c = -4
        d = 3 + 1e-8
        result = mkfs(a, b, c, d, f_shape=(3, 1))
        assert result.shape == (3, 4)

    def test_four(self):
        a = torch.randn(2, 2)
        b = torch.randn(2, 1)
        result = mkfs(a, b, f_shape=(4, 4))
        assert result.shape == (2, 3)

    def test_five(self):
        assert mkfs(3).shape == (1,)

    def test_six(self):
        assert mkfs(4, f_shape=(3, 4)).shape == (3, 4)

    def test_seven(self):
        a = torch.randn(15, 200, 2)
        b = torch.randn(15, 200, 1)
        c = 1.5
        result = mkfs(a, b, c)
        assert result.shape == (15, 200, 4)

        result = mkfs(c, a, b)
        assert result.shape == (15, 200, 4)
