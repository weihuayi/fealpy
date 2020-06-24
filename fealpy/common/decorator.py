from functools import wraps
import warnings


def deprecate(func):
    def dwarning(*args, **kwargs):

        return func(*args, **kwargs)
    return add_attribute 

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

