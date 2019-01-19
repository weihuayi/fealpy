from setuptools import setup
import platform 

s = platform.system()
if s is 'Linux':
    setup(name='fealpy',
          version='0.1',
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
              'cython',
              'pybind11',
              'msgpack',
              'PyHamcrest',
              'boost',
              'pytools',
              'meshpy',
              'pyamg'
          ],
          zip_safe=False)
elif s is 'Windows': 
    setup(name='fealpy',
          version='0.1',
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
elif s is 'Darwin': # Mac OS
    setup(name='fealpy',
          version='0.1',
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
else:
    print('I do not known your system type! Please contact weihuayi@xtu.edu.cn')
