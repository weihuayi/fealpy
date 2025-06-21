"""
sampler
=======
Provide samplers for training neural networks.
"""

from .collocator import (
    Collocator, CircleCollocator, LineCollocator, QuadrangleCollocator,
    PolygonCollocator, SphereCollocator
)
from .sampler import ISampler, BoxBoundarySampler