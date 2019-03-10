import numpy as np

class NumCompComponent():
    """
    并行计算构件， 负责局部计算和通信
    """
    def __init__(self, commtop):
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


    def communicating(self, array):
        ct = self.commtop
        comm = ct.comm
        rank = comm.Get_rank()
        for r in ct.neighbor: 
            data = array[ct.sds[r]]
            print("rank: ", rank, "data: ", len(ct.sds[r]), len(ct.rds[r]), "r: ", r)
            comm.Isend(data, dest=r, tag=rank) 

        for r in ct.neighbor:  
            data = np.zeros(len(ct.rds[r]), dtype=array.dtype)
            req = comm.Irecv(data, source=r, tag=r)
            req.Wait()
            array[ct.rds[r]] = data
