"""

Notes
-----
在这个模块中, 我们引入了 coordinates 修饰子,  可以给修饰对象加一个 coordtype 的
属性. 其它程序在调用该函数时, 可以用来决定传入参数的坐标类型.
"""
from functools import wraps

def cartesian(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'cartesian' 
    return add_attribute 

def barycentric(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'barycentric' 
    return add_attribute 

def polar(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'polar' 
    return add_attribute 

def spherical(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'spherical' 
    return add_attribute 

def cylindrical(func):
    @wraps(func)
    def add_attribute(*args, **kwargs):
        return func(*args, **kwargs)
    add_attribute.__dict__['coordtype'] = 'cylindrical' 
    return add_attribute 
