"""
sampler
=======
Provide samplers for training neural networks.
"""

from .sampler import Sampler, ConstantSampler, ISampler, BoxBoundarySampler
from .sampler import random_weights
from .sampler import TriangleMeshSampler, TetrahedronMeshSampler
