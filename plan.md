

#

使用 argparse 优化脚本的输入参数 

```
import argparse
    import argparse
    description = 'Read and display ExodusII data.'
    epilogue = '''
   '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue)
    parser.add_argument('filename', help='A required filename e.g mug.e.')
    parser.add_argument('nodal_var', help='The nodal variable e,g, convected.')
    args = parser.parse_args()
    return args.filename, args.nodal_var
```
# Don't use numpy's aliases of Python builtin objects. 
https://github.com/scipy/scipy/pull/12344/commits/02def703b8b7b28ed315a658808364fd024bb45c

np.int --> np.int_
np.bool --> np.bool_
np.float --> np.float64
np.complex --> np.complex128

# 2018.08.05

1. 更新线弹性的代码， 提升速度
2. 低秩化表示应力变量， 节省存储空间

# 2017.1.23

* coloring algorithm 
*
