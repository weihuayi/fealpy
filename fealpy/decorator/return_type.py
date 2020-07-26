
"""

Notes
-----
在这个模块中, 我们引入了函数返回类型的装饰器,  可以给装饰对象加一个 returntype 的
属性. 其它程序在调用该函数时, 可以用来决定传入参数的坐标类型.
"""
from functools import wraps

def scalar(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['returntype'] = 'scalar' 
    return add_attribute 

def vector(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['returntype'] = 'vector' 
    return add_attribute 

def matrix(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['returntype'] = 'matrix' 
    return add_attribute 

