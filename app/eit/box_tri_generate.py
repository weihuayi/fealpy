"""
此脚本用于生成电阻抗成像数据集。
求解区域为 [-1, 1], [-1, 1]。
"""

import os
from typing import Sequence
from time import time
import argparse

import numpy as np
import yaml
from tqdm import tqdm

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.backend import TensorLike as Tensor
from fealpy.experimental.mesh import TriangleMesh, UniformMesh2d
from fealpy.experimental import logger
from fealpy.cem import EITDataGenerator


logger.setLevel('INFO')
parser = argparse.ArgumentParser()
parser.add_argument("config", help="path of the .yaml file")


def levelset(p: Tensor, centers: Tensor, radius: Tensor):
    """Calculate level set function value."""
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = bm.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
    ret = bm.min(dis - radius[None, :], axis=-1) # (N, )
    return ret.reshape(struct)


# def levelset_np(p: NDArray, centers: NDArray, radius: NDArray):
#     """Calculate level set function value."""
#     struct = p.shape[:-1]
#     p = p.reshape(-1, p.shape[-1])
#     dis = np.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
#     ret = np.min(dis - radius[None, :], axis=-1) # (N, )
#     return ret.reshape(struct)


args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


EXT = config['data']['ext']
H = 2./EXT
SIGMA = config['data']['sigma']
NUM_CIR = config['data'].get('num_cir', 3)
FREQ = config['data']['freq']
DTYPE = getattr(bm, config['data']['dtype'])

BACKEND = config['data'].get('backend', None)
if BACKEND:
    bm.set_backend(BACKEND)
# DEVICE = config['data'].get('device', None)

if 'fem' in config:
    P = config['fem'].get('p', 1)
    Q = config['fem'].get('q', P + 2)
else:
    P, Q = 1, 3

# kwargs = {"dtype": DTYPE, "device": DEVICE}
output_folder = config['output_folder']
output_folder = os.path.join(output_folder, "")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'gd'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'inclusion'), exist_ok=True)


def neumann(points: Tensor, *args):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, 'device': bm.get_device(points)}
    theta = bm.arctan2(y, x)
    freq = bm.tensor(FREQ, **kwargs)
    return bm.sin(bm.tensordot(freq, theta, axes=0))


def main(sigma_iterable: Sequence[int], seed=0, index=0):
    np.random.seed(seed)
    umesh = UniformMesh2d([0, EXT, 0, EXT], [H, H], [-1., -1.], itype=bm.int32, ftype=DTYPE)

    for sigma_idx in tqdm(sigma_iterable,
                          desc=f"task{index}",
                          dynamic_ncols=True,
                          unit='sample',
                          position=index):

        ctrs_ = np.random.rand(NUM_CIR, 2) * 1.6 - 0.8 # (NCir, GD)
        b = np.min(0.9-np.abs(ctrs_), axis=-1) # (NCir, )
        rads_ = np.random.rand(NUM_CIR) * (b-0.1) + 0.1 # (NCir, )
        ctrs = bm.from_numpy(ctrs_)
        ctrs = bm.astype(ctrs, DTYPE)
        rads = bm.from_numpy(rads_)
        rads = bm.astype(rads_, DTYPE)

        ls_fn = lambda p: levelset(p, ctrs, rads)

        interface_mesh = TriangleMesh.interfacemesh_generator(umesh, phi=ls_fn)
        generator = EITDataGenerator(mesh=interface_mesh, p=P, q=Q)
        gn = generator.set_boundary(neumann, batch_size=len(FREQ))
        label = generator.set_levelset(SIGMA, ls_fn)
        gd = generator.run()

        np.save(
            os.path.join(output_folder, f'gd/{sigma_idx}.npy'),
            bm.to_numpy(gd)
        )
        np.savez(
            os.path.join(output_folder, f'inclusion/{sigma_idx}.npz'),
            label=bm.to_numpy(label),
            ctrs=bm.to_numpy(ctrs),
            rads=bm.to_numpy(rads)
        )

    if index == 0: # Save gn only on the first task
        np.save(os.path.join(output_folder, 'gn.npy'), bm.to_numpy(gn))

    return


def estimate_space_occupied(n_float: int, n_bool: int, n_samples: int, dtype: str):
    if dtype == bm.float32:
        single = n_float * 4 + n_bool
    elif dtype == bm.float64:
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

    NUM_PROCESS = config.get('num_process', 1)
    SEED_OFFSETS = config.get('seed_offsets', [0])
    NUM_CHUNK = len(SEED_OFFSETS)

    space_occ = estimate_space_occupied(EXT * 4 * len(FREQ),
                                        (EXT+1)**2,
                                        config['tail'] - config['head'],
                                        DTYPE)

    print("Start generating data...")
    print(f"using {NUM_PROCESS} processes, {NUM_CHUNK} data chunks.")
    print("Config:")
    print(f"    mesh: {EXT}x{EXT}, order: {P}, integral: {Q}")
    print(f"    sigma(inclusion): {SIGMA[0]}")
    print(f"    sigma(background): {SIGMA[1]}")
    print(f"    number of circles: {NUM_CIR}")
    print(f"    freq: {FREQ}")
    print(f"    dtype: {DTYPE}")
    print("Output:")
    print(f"    data index range: {config['head']}~{config['tail']}")
    print(f"    data shape: {2}x{EXT*4}")
    print(f"    n_channel: {len(FREQ)}")
    print(f"    label shape: {EXT+1}x{EXT+1}")
    print(f"Space occupation estimate: {_unit(space_occ)}.")
    print(f"All data will be saved to folder: {output_folder}", end='\n\n')
    signal_ = input("Continue? (y/n)")

    if signal_ not in {'y', 'Y'}:
        print("Aborted.")
        exit(0)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    from multiprocessing import Pool
    from fealpy.utils import timer

    pool = Pool(NUM_PROCESS)
    tmr = timer()
    next(tmr)

    SEED = config.get('seed', int(time()))
    NUM = tuple(range(config['head'], config['tail']))

    for idx, offset in enumerate(SEED_OFFSETS):
        pool.apply_async(main, (NUM[idx::NUM_CHUNK], offset + SEED, idx))

    pool.close()
    pool.join()

    tmr.send('stop')
    next(tmr)

    print("Done.")
