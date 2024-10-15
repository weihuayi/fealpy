import numpy as np

class CommToplogy():
    """
    通信拓扑数据结构类，目前有两个子类：
    1. 矩阵
    1. 网格 
    """
    def __init__(self, comm):
        self.comm = comm
        self.neighbor = None # 当前进程的邻居进程集合
        # 需要发送的数据编号信息，编号存储顺序应该是对方约定好的顺序
        self.sds = {} 
        # 需要接收的数据编号信息，
        self.rds = {} 

class CSRMatrixCommToplogy(CommToplogy):
    """CSRMatrixCommToplogy

    Note
    ----
    CSR 稀疏矩阵的通信拓扑数据结构，这里的实现假设CSR矩阵是按行分块的。在有限元
    等应用中，矩阵的分块是行列都进行分块的，需要另外一个类来实现。

    这里还假设当前进程只知道自己的行分块矩阵。当然还存在另外一种模式，就是一个主
    进程进行任务分割，并建立好通信拓扑，然后广播给每个进程。两种方式需要探讨优劣
    。
    """
    def __init__(self, comm, N):
        """__init__

        :param comm: 通信子
        :param    N: CSR 矩阵规模
        """
        super(CSRMatrixCommToplogy, self).__init__(comm) 

        comm = self.comm
        size = comm.Get_size()
        NN = N//size
        RE = N%size

        # 初始任务划分
        count = NN*np.ones(size, dtype='i')
        count[0:RE] += 1
        self.location = np.zeros(size+1, dtype='i')
        self.location[1:] = np.cumsum(count)

    def create_comm_toplogy(self, indices):
        """create_comm_toplogy

        创建通信拓扑

        Parameter
        ---------
        indices: 当前局部 CSR 矩阵的列指标数组

        Note
        ----
        1. 接收数据必须首先开辟好内存空间，这是必须的吗？
        """

        comm = self.comm
        size = comm.Get_size()
        rank = comm.Get_rank()

        indices = np.unique(indices)
        isNotLocal = (indices < self.location[rank]) | (indices >= self.location[rank+1])
        indices = indices[isNotLocal]

        N = self.location[-1]
        NN = N//size
        RE = N%size
        ranks = (indices - RE)//NN
        ranks[ranks<0] = 0
        self.neighbor = set(ranks)

        #发送当前进程需要接收数据的个数
        for r in self.neighbor:
            self.rds[r] = indices[ranks==r]
            data = np.array(len(self.rds[r]), dtype='i')
            comm.Isend(data, dest=r, tag=rank)

        #接收当前进程需要发送数据的个数，并分配内存
        for r in self.neighbor:
            data = np.zeros(1, dtype='i')
            req = comm.Irecv(data, source=r, tag=r)
            req.Wait()
            self.sds[r] = np.zeros(data[0], dtype='i') 

        #发送当前进程需要接收数据的编号信息
        for r in self.neighbor: 
            data = self.rds[r]
            comm.Isend(data, dest=r, tag=rank) 

        #接收当前进程需要发送数据的编号信息
        for r in self.neighbor:  
            data = self.sds[r]
            req = comm.Irecv(data, source=r, tag=r)
            req.Wait()

    def get_parallel_operator(self, A):
        rank = self.comm.Get_rank()
        A = A[self.location[rank]:self.location[rank+1]]
        self.create_comm_toplogy(A.indices)
        return A

    def get_local_idx(self):
        rank = self.comm.Get_rank()
        return np.arange(self.location[rank], self.location[rank+1], dtype='i')
