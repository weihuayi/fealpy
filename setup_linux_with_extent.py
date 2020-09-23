import os
from setuptools import Extension, setup

class get_pybind_include:
    """Helper class to determine the pybind11 include path. The purpose of this class is
    to postpone importing pybind11 until it is actually installed, so that the
    ``get_include()`` method can be invoked.
    """
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        "fealpy.extent.cgal",
        [
            "src/cgal/pybind11.cpp",
        ],
        language="C++",
        include_dirs=[
            os.environ.get("EIGEN_INCLUDE_DIR", "/usr/include/eigen3/"),
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        libraries=["stdc++", "gmp", "mpfr"],
    ),
    Extension(
        "fealpy.extent.detri2",
        [
            "src/detri2/pybind11.cpp",
            "src/detri2/detri2.cpp",
            "src/detri2/flips.cpp",
            "src/detri2/pred3d.cpp",
            "src/detri2/io.cpp",
            "src/detri2/sort.cpp",
            "src/detri2/delaunay.cpp",
            "src/detri2/constrained.cpp",
            "src/detri2/refine.cpp",
            "src/detri2/voronoi.cpp",
            "src/detri2/adapt.cpp",
            "src/detri2/metric.cpp",
        ],
        language="C++",
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        libraries=["gmp", "gmpxx"],
    )
]

if __name__ == "__main__":
    setup(name='fealpy',
          version='1.0',
          description='FEALPy: Finite Element Analysis Library in Python',
          url='http://github.com/weihuayi/fealpy',
          author='Huayi Wei',
          author_email='weihuayi@xtu.edu.cn',
          license='GNU',
          packages=['fealpy'],
          install_requires=[
              'numpy',
              'scipy',
              'matplotlib',
          ],
          ext_modules = ext_modules
      )
