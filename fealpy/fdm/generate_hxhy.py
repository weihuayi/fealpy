import numpy as np
import random

def Normal_random(m,n):
    x = random.uniform(1.2,9.9,m)
    y = random.uniform(2.7,17.94,n)
    hx = x/sum(x)
    hy = y/sum(y)
    return hx,hy

