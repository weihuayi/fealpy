from dask.distributed import Client


def inc(x):
    return x + 1

if __name__ == '__main__':
    client = Client()
    future = client.submit(inc, 10)

    result = future.result()
    print(result)  # 输出应为 11

    futures = [client.submit(inc, i) for i in range(10)]
    results = client.gather(futures)
    print(results)  # 输出应为 [1, 2, 3, ..., 10]


