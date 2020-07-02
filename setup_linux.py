from setuptools import setup
import os

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
          'pyamg',
          'meshpy',
          'pyfftw'
      ],
      zip_safe=False)
