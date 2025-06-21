
import os
from setuptools import Extension

def update(ext_modules):
    ext_modules.append(
        Extension(
            'fealpy.solver.mumps._dmumps',
            sources=['fealpy/solver/mumps/_dmumps.pyx'],
            libraries=['dmumps', 'mumps_common'],
            )
    )

def build(dep, ext_modules):
    update(ext_modules)
