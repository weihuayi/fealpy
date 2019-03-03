import numpy as np

class NumCompComponent():
    """
    并行计算构件， 负责局部计算和通信
    """
    def __init__(self, commtop, numop):
        """
        初始化函数

        Parameter
        ---------
        commtop: 通信拓扑对象
            存储通信相关的数据
        numop: 数值计算对象
            进行局部的数值计算

        Note
        ----
        """
        self.commtop = commtop
        self.numop = numop 

    def communicating(self, parray):
        ct = self.commtop
        comm = ct.comm
        for r in ct.neighbor: 
            data = parray[self.sendds[r]]
            comm.Isend(data, dest=r, tag=rank) 
        for r in ct.neighbor:  
            data = np.zeros(len(self.rds[r]), dtype=parray.dtype)
            req = comm.Irecv(data, source=r, tag=r)
            req.Wait()
            parray[self.recvds[r]] = data
