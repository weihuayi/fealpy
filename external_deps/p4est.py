
import os
from setuptools import Extension

def update(ext_modules):
    ext_modules.append(
        Extension(
            'fealpy.mesh.p4est._p4est',
            sources=['fealpy/mesh/p4est/_p4est.pyx'],
            libraries=['p4est'],
            include_dirs=['/usr/lib/x86_64-linux-gnu/openmpi/include']
        )
    )

def build(dep, ext_modules):
    update(ext_modules)
