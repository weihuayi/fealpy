#!/usr/bin/env python3
# 
import sys 
import numpy as np

from fealpy.common import DynamicArray


class DynamicArrayTest():
    def __init__(self):
        pass

    def append(self):
        a = np.arange(10*3).reshape(10, 3)
        b = DynamicArray(a, capacity=100)
        c = np.arange(80*3).reshape(80, 3)
        b.extend(c)
        print(b)

    def index(self):
        a = np.arange(10*3).reshape(10, 3)
        b = DynamicArray(a, capacity=100)
        print(b[7:])



test = DynamicArrayTest()

if sys.argv[1] == 'append':
    test.append()

if sys.argv[1] == 'index':
    test.index()
