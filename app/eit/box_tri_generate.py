"""
此脚本用于生成数据集，每个样本包含多个 gD & gN 数据通道和标签。
求解区域为 [-1, 1], [-1, 1]。
"""

import os
from typing import Sequence
from time import time
import argparse

import torch
from torch import Tensor, tensordot, rand
import numpy as np
import yaml
from tqdm import tqdm
# from viztracer import VizTracer

from fealpy.torch.mesh import TriangleMesh
from fealpy.torch import logger

from data_generator import EITDataGenerator

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


args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


EXT = config['data']['ext']
SIGMA = config['data']['sigma']
NUM_CIR = config['data'].get('num_cir', 3)
FREQ = config['data']['freq']
DTYPE = getattr(torch, config['data']['dtype'])
DEVICE = config['data'].get('device', None)
output_folder = config['output_folder']
kwargs = {"dtype": DTYPE, "device": DEVICE}
os.path.join(output_folder, "")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def neumann(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    theta = torch.arctan2(y, x)
    freq = torch.tensor(FREQ, **kwargs)
    return torch.sin(tensordot(freq, theta, dims=0))


def main(sigma_iterable: Sequence[int], seed=0, index=0):
    torch.manual_seed(seed)
    mesh = TriangleMesh.from_box((-1, 1, -1, 1), EXT, EXT, ftype=DTYPE, device=DEVICE)
    generator = EITDataGenerator(mesh=mesh)
    gn = generator.set_boundary(neumann, batched=True)
    if index == 0: # Save gn only on the first task
        np.save(os.path.join(output_folder, 'gn.npy'), gn.cpu().numpy())

    for sigma_idx in tqdm(sigma_iterable,
                          desc=f"task{index}",
                          dynamic_ncols=True,
                          unit='sample',
                          position=index):
        ctrs = rand(NUM_CIR, 2, **kwargs) * 1.6 - 0.8 # (NCir, GD)
        b, _ = torch.min(0.9-torch.abs(ctrs), axis=-1) # (NCir, )
        rads = rand(NUM_CIR, **kwargs) * (b-0.1) + 0.1 # (NCir, )
        ls_fn = lambda p: levelset(p, ctrs, rads)

        label = generator.set_levelset(SIGMA, ls_fn)
        gd = generator.run()
        np.savez(
            os.path.join(output_folder, f'gd_{sigma_idx}.npz'),
            gd=gd.numpy(),
            label=label.cpu().numpy(),
            ctrs=ctrs.cpu().numpy(),
            rads=rads.cpu().numpy()
        )


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
    space_occ = estimate_space_occupied(EXT * 8 * len(FREQ),
                                        (EXT+1)**2,
                                        config['tail'] - config['head'],
                                        DTYPE)

    print("Start generating data...")
    print(f"using {process_num} processes")
    print("Config:")
    print(f"    mesh: {EXT}x{EXT}")
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
    from fealpy.ml import timer

    pool = Pool(process_num)
    tmr = timer()
    tmr.send(None)

    NUM = tuple(range(config['head'], config['tail']))

    PART = 4
    TM = int(time())
    # tracer = VizTracer()
    # tracer.start()
    # main(NUM, 999, 0)
    # tracer.stop()
    # tracer.save(f'{output_folder}/{TM}.json')

    pool.apply_async(main, (NUM[0::PART], 621 + TM, 0))
    pool.apply_async(main, (NUM[1::PART], 928 + TM, 1))
    pool.apply_async(main, (NUM[2::PART], 122 + TM, 2))
    pool.apply_async(main, (NUM[3::PART], 222 + TM, 3))

    pool.close()
    pool.join()

    tmr.send('stop')

    print("Done.")
