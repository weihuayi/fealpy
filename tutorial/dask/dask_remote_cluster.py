# 在命令行中启动调度器和工作节点
# $ dask-scheduler
# $ dask-worker tcp://<scheduler-ip>:8786 
# 
# 监控集群状态：
# 任务：启动集群后，打开浏览器并访问 http://<scheduler-ip>:8787，查看集群的状态和任务执行情况。

from dask.distributed import Client

def inc(x):
    return x + 1

if __name__ == '__main__':
    client = Client('tcp://<scheduler-ip>:8786')

    future = client.submit(inc, 10)
    result = future.result()
    print(result)  # 输出应为 11

