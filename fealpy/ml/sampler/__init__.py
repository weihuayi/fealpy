"""
sampler
=======
Provide samplers for training neural networks.
"""

from .sampler import Sampler, ConstantSampler, ISampler, BoxBoundarySampler
from .sampler import MeshSampler
from .collocator import (
    Collocator, CircleCollocator, LineCollocator, QuadrangleCollocator,
    PolygonCollocator, SphereCollocator
)
from .interface import InterfaceSampler
