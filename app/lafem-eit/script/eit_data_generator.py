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

from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike as Tensor
from fealpy.mesh import TriangleMesh, UniformMesh2d
from fealpy.cem import EITDataGenerator


parser = argparse.ArgumentParser()
parser.add_argument("config", help="path of the .yaml file")


def levelset(p: Tensor, centers: Tensor, radius: Tensor):
    """Calculate level set function value."""
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = bm.linalg.norm(p[:, None, :] - centers[None, :, :], axis=-1) # (N, NCir)
    ret = bm.min(dis - radius[None, :], axis=-1) # (N, )
    return ret.reshape(struct)

def transition_sine(levelset_val: Tensor, /, width: float) -> Tensor:
    PI = bm.pi
    coef = PI / width
    val = bm.clip(levelset_val*coef, -PI/2, PI/2)
    return 0.5 - bm.sin(val) * 0.5


args = parser.parse_args()
with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

### Read Args ###

GD_FOLDER = config['data'].get('gd_folder', 'gd')
GN_FILE = config['data'].get('gn_file', 'gn')
EXT = config['data']['ext']
H = 2./EXT
SIGMA = config['data']['sigma']
NUM_CIR = config['data'].get('num_cir', 3)
FREQ = config['data']['freq']
LABEL_FOLDER = config['label'].get('label_folder', 'inclusion')
LABEL_DTYPE = getattr(bm, config['label']['dtype'])
LABEL_TRANS = config['label'].get('transition', 'bool')

if LABEL_TRANS == 'bool':
    trans = lambda x: x < 0.
elif LABEL_TRANS == 'sine':
    trans = lambda x: transition_sine(x, 2*H)
else:
    raise ValueError

BACKEND = config['data'].get('backend', None)
if BACKEND:
    bm.set_backend(BACKEND)

if 'fem' in config:
    P = config['fem'].get('p', 1)
    Q = config['fem'].get('q', P + 2)
else:
    P, Q = 1, 3

output_folder = config['output_folder']
output_folder = os.path.join(output_folder, "")
os.makedirs(output_folder, exist_ok=True)

if GD_FOLDER:
    os.makedirs(os.path.join(output_folder, GD_FOLDER), exist_ok=True)
if LABEL_FOLDER:
    os.makedirs(os.path.join(output_folder, LABEL_FOLDER), exist_ok=True)


def neumann(points: Tensor, *args):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, 'device': bm.get_device(points)}
    theta = bm.arctan2(y, x)
    freq = bm.tensor(FREQ, **kwargs)
    return bm.sin(bm.tensordot(freq, theta, axes=0))


def main(sigma_iterable: Sequence[int], seed: int = 0, index: int = 0):
    np.random.seed(seed)
    umesh = UniformMesh2d([0, EXT, 0, EXT], [H, H], [-1., -1.], itype=bm.int32, ftype=bm.float64)
    pixel = umesh.entity('node')

    for sigma_idx in tqdm(sigma_iterable,
                          desc=f"task{index}",
                          dynamic_ncols=True,
                          unit='sample',
                          position=index):

        ctrs_ = np.random.rand(NUM_CIR, 2) * 1.6 - 0.8 # (NCir, GD)
        b = np.min(0.9-np.abs(ctrs_), axis=-1) # (NCir, )
        rads_ = np.random.rand(NUM_CIR) * (b-0.1) + 0.1 # (NCir, )
        ctrs = bm.astype(ctrs_, bm.float64)
        rads = bm.astype(rads_, bm.float64)

        ls_fn = lambda p: levelset(p, ctrs, rads)

        interface_mesh = TriangleMesh.interfacemesh_generator(umesh, phi=ls_fn)
        generator = EITDataGenerator(mesh=interface_mesh, p=P, q=Q)
        gn = generator.set_boundary(neumann, batch_size=len(FREQ))
        generator.set_levelset(SIGMA, ls_fn)
        gd = generator.run()

        label = bm.astype(trans(ls_fn(pixel)), LABEL_DTYPE)

        if GD_FOLDER:
            np.save(
                os.path.join(output_folder, f'{GD_FOLDER}/{sigma_idx}.npy'),
                bm.to_numpy(gd)
            )
        if LABEL_FOLDER:
            np.savez(
                os.path.join(output_folder, f'{LABEL_FOLDER}/{sigma_idx}.npz'),
                label=bm.to_numpy(label),
                ctrs=bm.to_numpy(ctrs),
                rads=bm.to_numpy(rads)
            )

    if index == 0 and GN_FILE: # Save gn only on the first task
        np.save(os.path.join(output_folder, f'{GN_FILE}.npy'), bm.to_numpy(gn))

    return


def estimate_space_occupied(n_samples: int, data_fp: int, label: int, label_dtype):
    if label_dtype == bm.bool:
        single = data_fp * 8 + label
    elif label_dtype == bm.float32:
        single = data_fp * 8 + label * 4
    elif label_dtype == bm.float64:
        single = data_fp * 8 + label * 8
    else:
        print(f"Unsupported label dtype {label_dtype} to estimate space occupation.")
        return 0.0
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

    space_occ = estimate_space_occupied(
        config['tail'] - config['head'],
        EXT * 4 * len(FREQ) if GD_FOLDER else 0,
        (EXT+1)**2 if LABEL_FOLDER else 0,
        LABEL_DTYPE)

    print("Start generating data...")
    print(f"    using {NUM_PROCESS} processes, {NUM_CHUNK} data chunks.")
    print("Config:")
    print(f"    mesh: {EXT}x{EXT}, order: {P}, integral: {Q}")
    print(f"    sigma(inclusion/background): {SIGMA[0], SIGMA[1]}")
    print(f"    number of circles: {NUM_CIR}")
    print(f"    freq: {FREQ}")
    print(f"    label dtype: {LABEL_DTYPE}")
    print(f"    label transition: {LABEL_TRANS}")
    print("Output:")
    print(f"    data index range: {config['head']}~{config['tail']}")
    print(f"    STRUCTURE: {output_folder}")
    if GD_FOLDER: print(f"    -- {GD_FOLDER}/*.npy")
    if GN_FILE: print(f"    -- {GN_FILE}.npy")
    if LABEL_FOLDER: print(f"    -- {LABEL_FOLDER}/*.npz")
    print(f"Space occupation estimate: {_unit(space_occ)}.\n\n")
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
