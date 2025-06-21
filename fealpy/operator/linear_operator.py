

class LinearOperator:

    def __init__(self, shape, matvec):
        self.__matvec_impl = matvec


    def _matvec(self, x):
        return self.__matvec_impl(x)

    def __matmul__(self, x):
        return self.__matvec_impl(x)