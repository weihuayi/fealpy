"""
sampler
=======
Provide samplers for training neural networks.
"""

from .sampler import Sampler, ConstantSampler, ISampler, BoxBoundarySampler
from .sampler import random_weights
from .sampler import (
    get_mesh_sampler,
    TMeshSampler,
    TriangleMeshSampler,
    TetrahedronMeshSampler,
    QuadrangleMeshSampler
)
from .collocator import Collocator, CircleCollocator, LineCollocator, QuadrangleCollocator
