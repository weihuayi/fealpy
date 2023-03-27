
import numpy as np
import torch
import torch.nn as nn

from fealpy.pinn.machine import Solution
from fealpy.mesh import MeshFactory as Mf


class TestMachine():
    def test_basic(self):
        net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 16)
        )
        s = Solution(net)

        ps = torch.empty((15, 200, 2), dtype=torch.float32)
        assert s(ps).shape == (15, 200, 16)

        mesh = Mf.boxmesh2d([0, 1, 0, 1], nx=10, ny=10)
        bcs = np.empty((100, 3), dtype=np.float32)
        assert s.from_cell_bc(bcs, mesh).shape == (100, 200, 16)

    def test_fix(self):
        s = Solution(nn.Linear(4, 2))
        px = torch.empty((100, 1), dtype=torch.float32)
        py = torch.empty((100, 1), dtype=torch.float32)
        s1 = s.fixed([0, 2], [10, 20])
        assert s1(px, py).shape == (100, 2)

    def test_extracted(self):
        s = Solution(nn.Linear(2, 3))
        ps = torch.empty((100, 2), dtype=torch.float32)
        s1 = s.extracted(0, 2)
        assert s1(ps).shape == (100, 2)

    def test_mkfs(self):
        s = Solution(nn.Linear(10, 2))
        px = torch.empty((100, 3), dtype=torch.float32)
        py = torch.empty((100, 2), dtype=torch.float32)
        pz = torch.empty((100, 2), dtype=torch.float32)
        ret = s(0, px, py, 1, pz, 0)
        assert ret.shape == (100, 2)
