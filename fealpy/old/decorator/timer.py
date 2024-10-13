"""

Notes
-----
在这个模块中, 我们引入了时间统计的装饰子
"""

from functools import wraps
from timeit import default_timer as dtimer 

def timer(func):
    """
    Notes
    -----
    测试函数运行的墙上时间。
    """
    @wraps(func)
    def run(*args, **kwargs):
        start = dtimer()
        val = func(*args, **kwargs)
        end = dtimer()
        print('run {} with time:'.format(func.__name__), end - start)
        return val 
    return run 
