"""
此脚本用于生成电阻抗成像数据集。
求解区域为 [-1, 1], [-1, 1]。
"""

import os
from typing import Sequence
from time import time
import argparse

import torch
from torch import Tensor, tensordot, rand
import numpy as np
from numpy.typing import NDArray
import yaml
from tqdm import tqdm

from fealpy.mesh import TriangleMesh as TMD
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch import logger
from fealpy.cem import EITDataGenerator


logger.setLevel('WARNING')
parser = argparse.ArgumentParser()
parser.add_argument("config", help="path of the .yaml file")


def levelset(p: Tensor, centers: Tensor, radius: Tensor):
    """
    Calculate level set function value.
    """
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = torch.norm(p[:, None, :] - centers[None, :, :], dim=-1) # (N, NCir)
    ret, _ = torch.min(dis - radius[None, :], dim=-1) # (N, )
    return ret.reshape(struct)


def levelset_np(p: NDArray, centers: NDArray, radius: NDArray):
    """
    Calculate level set function value.
    """
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
    ret = np.min(dis - radius[None, :], axis=-1) # (N, )
    return ret.reshape(struct)


args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


EXT = config['data']['ext']
SIGMA = config['data']['sigma']
NUM_CIR = config['data'].get('num_cir', 3)
FREQ = config['data']['freq']
DTYPE = getattr(torch, config['data']['dtype'])
DEVICE = config['data'].get('device', None)

if 'fem' in config:
    P = config['fem'].get('p', 1)
    Q = config['fem'].get('q', P + 2)
else:
    P, Q = 1, 3

output_folder = config['output_folder']
kwargs = {"dtype": DTYPE, "device": DEVICE}
os.path.join(output_folder, "")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, f'gd'), exist_ok=True)
os.makedirs(os.path.join(output_folder, f'inclusion'), exist_ok=True)


def neumann(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    theta = torch.arctan2(y, x)
    freq = torch.tensor(FREQ, **kwargs)
    return torch.cos(tensordot(freq, theta, dims=0))


def main(sigma_iterable: Sequence[int], seed=0, index=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mesh = TriangleMesh.from_box((-1, 1, -1, 1), EXT, EXT, ftype=DTYPE, device=DEVICE)
    generator = EITDataGenerator(mesh=mesh, p=P, q=Q)
    gn = generator.set_boundary(neumann, batch_size=len(FREQ))

    if index == 0: # Save gn only on the first task
        np.save(os.path.join(output_folder, 'gn.npy'), gn.cpu().numpy())

    for sigma_idx in tqdm(sigma_iterable,
                          desc=f"task{index}",
                          dynamic_ncols=True,
                          unit='sample',
                          position=index):
        # ctrs = rand(NUM_CIR, 2, **kwargs) * 1.6 - 0.8 # (NCir, GD)
        # b, _ = torch.min(0.9-torch.abs(ctrs), axis=-1) # (NCir, )
        # rads = rand(NUM_CIR, **kwargs) * (b-0.1) + 0.1 # (NCir, )

        ctrs_ = np.random.rand(NUM_CIR, 2) * 1.6 - 0.8 # (NCir, GD)
        b = np.min(0.9-np.abs(ctrs_), axis=-1) # (NCir, )
        rads_ = np.random.rand(NUM_CIR) * (b-0.1) + 0.1 # (NCir, )
        ctrs = torch.from_numpy(ctrs_).to(**kwargs)
        rads = torch.from_numpy(rads_).to(**kwargs)

        ls_fn_np = lambda p: levelset_np(p, ctrs_, rads_)
        ls_fn = lambda p: levelset(p, ctrs, rads)

        interface_mesh = TMD.interfacemesh_generator([-1, 1, -1, 1], EXT, EXT, ls_fn_np)
        interface_mesh = TriangleMesh.from_numpy(interface_mesh)
        interface_mesh.to(device=DEVICE)

        generator = EITDataGenerator(mesh=interface_mesh, p=P, q=Q)
        generator.set_boundary(neumann, batch_size=len(FREQ))
        generator.set_levelset(SIGMA, ls_fn)

        # use the uniform triangle mesh nodes to generate labels
        node = mesh.entity('node')
        label = levelset(node, ctrs, rads) < 0.

        gd = generator.run()

        np.save(
            os.path.join(output_folder, f'gd/{sigma_idx}.npy'),
            gd.numpy()
        )
        np.savez(
            os.path.join(output_folder, f'inclusion/{sigma_idx}.npz'),
            label=label.cpu().numpy(),
            ctrs=ctrs.cpu().numpy(),
            rads=rads.cpu().numpy()
        )

    return


def estimate_space_occupied(n_float: int, n_bool: int, n_samples: int, dtype: str):
    if dtype == torch.float32:
        single = n_float * 4 + n_bool
    elif dtype == torch.float64:
        single = n_float * 8 + n_bool
    else:
        raise ValueError(f"Unsupported dtype '{DTYPE}'.")
    return n_samples * single


def _unit(bytes: int):
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(bytes) / np.log(1024)))
    if i == 0:
        return f"{bytes:.2f} {size_name[i]}"
    else:
        return f"{bytes / 1024 ** i:.2f} {size_name[i]}"


if __name__ == "__main__":
    import os

    process_num = config['process_num']
    space_occ = estimate_space_occupied(EXT * 4 * len(FREQ),
                                        (EXT+1)**2,
                                        config['tail'] - config['head'],
                                        DTYPE)

    print("Start generating data...")
    print(f"using {process_num} processes")
    print("Config:")
    print(f"    mesh: {EXT}x{EXT}, order: {P}, integral: {Q}")
    print(f"    sigma(inclusion): {SIGMA[0]}")
    print(f"    sigma(background): {SIGMA[1]}")
    print(f"    number of circles: {NUM_CIR}")
    print(f"    freq: {FREQ}")
    print(f"    dtype: {DTYPE}")
    print("Output:")
    print(f"    data shape: {2}x{EXT*4}")
    print(f"    n_channel: {len(FREQ)}")
    print(f"    label shape: {EXT+1}x{EXT+1}")
    print(f"    data index range: {config['head']}~{config['tail']}")
    print(f"Space occupation estimate: {_unit(space_occ)},")
    print(f"will be saved to folder: {output_folder}", end='\n\n')
    signal_ = input("Continue? (y/n)")

    if signal_ not in {'y', 'Y'}:
        print("Aborted.")
        exit(0)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    from multiprocessing import Pool
    from fealpy.utils import timer

    pool = Pool(process_num)
    tmr = timer()
    next(tmr)

    NUM = tuple(range(config['head'], config['tail']))

    PART = 4
    seed = config.get('seed', int(time()))
    # main(NUM, 999, 0)

    pool.apply_async(main, (NUM[0::PART], 621 + seed, 0))
    pool.apply_async(main, (NUM[1::PART], 928 + seed, 1))
    pool.apply_async(main, (NUM[2::PART], 122 + seed, 2))
    pool.apply_async(main, (NUM[3::PART], 222 + seed, 3))

    pool.close()
    pool.join()

    tmr.send('stop')
    next(tmr)

    print("Done.")
