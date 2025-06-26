from functools import wraps
from ..backend.base import TensorLike

def multi_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 判断只有一个参数传入，并且它是一个列表，且其中的每个元素也是列表或元组
        if len(args) == 1 and isinstance(args[0], (list, tuple, TensorLike)) and all(isinstance(item, (list, tuple, TensorLike)) for item in args[0]):
            results = []
            param_sets = args[0]
            if isinstance(args[0], TensorLike):
                param_sets = param_sets.tolist()
            for idx, param_set in enumerate(param_sets):
                # param_set 应该是一个列表或元组，使用 *param_set 来解包调用原函数
                kwarg = {}
                for k, v in kwargs.items():
                    val = v[idx]
                    if isinstance(val, TensorLike):
                        val = val.tolist()
                    kwarg[k] = val
                results.append(func(*param_set, **kwarg))

            return results
        # 如果都不是序列，直接调用
        return func(*args, **kwargs)
    return wrapper