from setuptools import setup
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
          'matplotlib'
      ],
      zip_safe=False)
