import numpy
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

randNum = numpy.zeros(2)
if rank == 1:
        randNum = numpy.random.random_sample(2)
        print("Process", rank, "drew the number", randNum)
        comm.Isend(randNum, dest=0)
        req = comm.Irecv(randNum, source=0)
        req.Wait()
        print("Process", rank, "received the number", randNum)
if rank == 0:
        print("Process", rank, "before receiving has the number", randNum)
        req = comm.Irecv(randNum, source=1)
        req.Wait()
        print("Process", rank, "received the number", randNum)
        randNum *= 2
        comm.Isend(randNum, dest=1)
