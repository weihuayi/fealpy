#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test_jax.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Wed 24 Jul 2024 10:33:11 AM CST
	@bref 
	@ref 
''' 

import jax.numpy as jnp
from fealpy.jax.mesh.node_mesh import NodeMesh
from fealpy.jax.sph import SPHSolver, SiEulerAdvance
from fealpy.jax.sph import partition 
from fealpy.jax.sph.jax_md.partition import Sparse
from jax_md.util import Array
from jax_md import space
from jax import jit

dx = 0.02
box_size = jnp.array([1.0,1.0]) #模拟区域

mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)

displacement, shift = space.periodic(side=box_size) 

neighbor_fn = partition.neighbor_list(
    displacement,
    box_size,
    r_cutoff=3*dx,
    backend="jaxmd_vmap",
    capacity_multiplier=1.25,
    mask_self=False,
    format=Sparse,
    num_particles_max=mesh.nodedata["position"].shape[0],
    num_partitions=mesh.nodedata["position"].shape[0],
    pbc=[True, True, True],
)

result = SiEulerAdvance(0.1, mesh.nodedata)
