
import sys
import multiprocessing as mp
import numpy as np
from timeit import default_timer as dtimer 

def f(x):
    return sum(a[x[0]:x[1]]) 

if __name__ == '__main__':
    nc = mp.cpu_count()
    print(nc)

    n = int(sys.argv[1])
    NS = 100000000
    a = np.arange(NS)

    block = NS//n
    r = NS%n
    index = np.full(n+1, block)
    index[0] = 0
    index[1:r+1] += 1
    np.cumsum(index, out=index)

    start = dtimer()
    with mp.Pool(10) as p:
        print(p.map(f, zip(index[0:-1], index[1:])))

    end  = dtimer()

    print('time:', end - start)
