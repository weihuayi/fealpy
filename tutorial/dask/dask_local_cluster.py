from dask.distributed import Client, LocalCluster

def square(x):
    return x ** 2

if __name__ == '__main__':
    cluster = LocalCluster()
    client = Client(cluster)

    future = client.submit(square, 10)
    result = future.result()
    print(result)  # 输出应为 100

