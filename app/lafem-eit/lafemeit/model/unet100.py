
import os
import sys
from typing import Callable, Tuple, Optional, Union

import torch
from torch import Tensor, float32, device
import torch.nn as nn
from fealpy.mesh import TriangleMesh

sys.path.append("./src")

from .fractional_operator import StackedFractional, RegressiveFractional, Fractional, MultiChannelFractional
from .data_feature import LaplaceFEMSolver, DataPreprocessor, DataFeature


class ConvBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel: int, dtype=float32) -> None:
        super().__init__()
        in_, out_ = in_channel, out_channel
        self.conv_1 = nn.Conv2d(in_, out_, kernel, padding=kernel//2, dtype=dtype) # [N, 10, 64, 64]
        self.conv_2 = nn.Conv2d(out_, out_, kernel, padding=kernel//2, bias=False, dtype=dtype) # [N, 10, 64, 64]
        self.bn = nn.BatchNorm2d(out_, momentum=0.01, dtype=dtype)
        self.down = nn.AvgPool2d(kernel_size=2) # [N, 10, 32, 32]

    def forward(self, phi: Tensor):
        phi = self.conv_2(self.conv_1(phi))
        out = self.down(torch.tanh_(self.bn(phi)))
        return out, phi

    __call__: Callable[[Tensor], Tuple[Tensor, Tensor]]


class ConvTBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel: int, dtype=float32) -> None:
        super().__init__()
        in_, out_ = in_channel, out_channel
        self.up = nn.ConvTranspose2d(in_, in_//2, 3, 2, 1, 1, dtype=dtype)
        self.convt_1 = nn.ConvTranspose2d(in_, out_, kernel, padding=kernel//2, dtype=dtype)
        self.convt_2 = nn.ConvTranspose2d(out_, out_, kernel, padding=kernel//2, bias=False, dtype=dtype)
        self.bn = nn.BatchNorm2d(out_, momentum=0.01, dtype=dtype)

    def forward(self, phi: Tensor, conn: Tensor):
        phi = self.up(phi)
        phi = torch.cat([phi, conn], dim=1)
        del conn
        phi = self.convt_2(self.convt_1(phi))
        out = torch.tanh_(self.bn(phi))
        return out

    __call__: Callable[[Tensor, Tensor], Tensor]


class Unet(nn.Module):
    def __init__(self, n_channel: int, *, dtype=float32) -> None:
        super().__init__()

        self.cb1 = ConvBlock(n_channel, 12, 9, dtype=dtype) # [N, 12, 32, 32]
        self.cb2 = ConvBlock(12, 24, 5, dtype=dtype) # [N, 24, 16, 16]
        self.cb3 = ConvBlock(24, 48, 3, dtype=dtype) # [N, 48, 8, 8]
        self.cb4 = ConvBlock(48, 96, 3, dtype=dtype) # [N, 96, 4, 4]

        self.btm = nn.Conv2d(96, 192, 3, 1, 1, dtype=dtype)

        self.ctb4 = ConvTBlock(192, 96, 3, dtype=dtype) # [N, 96, 4, 4]
        self.ctb3 = ConvTBlock(96, 48, 3, dtype=dtype) # [N, 48, 16, 16]
        self.ctb2 = ConvTBlock(48, 24, 5, dtype=dtype) # [N, 24, 32, 32]
        self.ctb1 = ConvTBlock(24, 12, 9, dtype=dtype) # [N, 12, 64, 64]

        self.conv = nn.ConvTranspose2d(12, 1, 1, dtype=dtype)

    def forward(self, input: Tensor):

        phi, p1 = self.cb1(input)
        phi, p2 = self.cb2(phi)
        phi, p3 = self.cb3(phi)
        phi, p4 = self.cb4(phi)

        phi = self.btm(phi)

        phi = self.ctb4(phi, p4)
        del p4
        phi = self.ctb3(phi, p3)
        del p3
        phi = self.ctb2(phi, p2)
        del p2
        phi = self.ctb1(phi, p1)
        del p1
        phi = self.conv(phi)

        return phi

    __call__: Callable[[Tensor], Tensor]


class EITModel(nn.Module):
    def __init__(self, n_channel: int, mesh: TriangleMesh, frac: nn.Module,
                 *, network_dtype=float32) -> None:
        super().__init__()
        self.n_channel = n_channel
        solver = LaplaceFEMSolver(mesh, p=1)
        self.df_prepor = DataPreprocessor(solver)
        self.df_solver = DataFeature(solver, bc_filter=frac) # [N, 8, 64*64]
        self.bn = nn.BatchNorm2d(n_channel, momentum=0.01, dtype=mesh.ftype)
        self.coordinate = mesh.entity('node').reshape(64, 64, 2).permute(2, 0, 1) # [2, 64, 64]
        self.unet = Unet(n_channel+2, dtype=network_dtype)
        self.network_dtype = network_dtype

    def forward(self, input: Tensor):
        N = input.shape[0]
        coor = self.coordinate[None, ...].repeat(N, 1, 1, 1)
        gnvn = self.df_prepor(input)
        del input
        phi = self.df_solver(gnvn).reshape(N, self.n_channel, 64, 64)
        phi = self.bn(phi)
        phi = torch.cat([phi, coor], dim=1)
        del coor
        phi = phi.to(self.network_dtype)
        phi = self.unet(phi)
        phi = torch.sigmoid_(phi)
        return phi

    __call__: Callable[[Tensor], Tensor]


def build_eit_model(
        name: str,
        ext: int,
        n_channel: int, *,
        tag: str,
        fractype: str,
        eigen_file: str,
        ckpts_path: Optional[str] = None,
        device: Union[device, str, None] = None,
        verbose: bool = False
    ):
    BdDofs = ext * 4
    mesh = TriangleMesh.from_box([-1, 1, -1, 1], ext, ext, device=device)

    if fractype == 'nograd':
        frac = Fractional(BdDofs, device=device)
        frac.gamma.requires_grad_(False)
    elif fractype == 'single':
        frac = Fractional(BdDofs, device=device)
    elif fractype == 'multi':
        frac = MultiChannelFractional(BdDofs, n_channel, device=device)
    elif fractype == 'stack':
        frac = StackedFractional(BdDofs, n_channel, device=device)
    elif fractype == 'reg':
        frac = RegressiveFractional(BdDofs, n_channel, device=device)
    else:
        raise NotImplementedError(f'Unknown type: {fractype}')

    frac.from_npz(eigen_file)
    model = EITModel(n_channel, mesh, frac, network_dtype=float32)
    model.to(device)

    FULL_NAME = (name + '_' + tag) if tag else name

    if verbose:
        print(f"Model built: {FULL_NAME}, in device: {device}")

        n_p = sum(p.numel() for p in model.unet.parameters())
        print(f"Number of parameters: {n_p/1e6:.2f}M")

    if ckpts_path is not None:
        try:
            ckpts_path = os.path.join(ckpts_path, f"{FULL_NAME}.pth")
            model.load_state_dict(torch.load(ckpts_path, map_location=device))
            if verbose: print(f"INFO: Checkpoint loaded from {ckpts_path}")
        except FileNotFoundError:
            pass

    return model, FULL_NAME


if __name__ == "__main__":
    build_eit_model('cpu', '')
