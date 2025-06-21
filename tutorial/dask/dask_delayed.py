from dask import delayed

@delayed
def inc(x):
    return x + 1

@delayed
def add(x, y):
    return x + y

x = inc(1)
y = inc(2)
z = add(x, y)

# 延迟执行，构建计算图
print(z)  # 输出 Delayed 对象

result = z.compute()
print(result)  # 输出应为 5

a = inc(1)
b = inc(2)
c = add(a, b)
d = inc(c)

result = d.compute()
print(result)  # 输出应为 6

