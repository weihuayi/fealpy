

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

import timeit, functools

t = timeit.Timer(functools.partial(fib, 20))
print(t.timeit(5))


fib = memoize(fib)
t = timeit.Timer(functools.partial(fib, 40))
print(t.timeit(5))
