
import torch

from fealpy.torch.sparse.linalg import ichol_coo


def test_sp_ichol_coo():
    i = torch.tensor([[0, 1, 0, 1, 2, 3, 0, 3, 1, 3],
                      [0, 0, 1, 1, 2, 0, 3, 1, 3, 3]])
    v = torch.tensor([9., 2., 2., 5., 6., 1., 1., 3., 3., 9.])
    A_sparse = torch.sparse_coo_tensor(i, v, (4, 4))

    ni, nv = ichol_coo(i, v, (4, 4))

    Ad = A_sparse.to_dense()
    Ld = torch.sparse_coo_tensor(ni, nv, (4, 4)).to_dense()

    print(Ad)
    print(Ld)
    print(Ld@Ld.T)


if __name__ == '__main__':
    test_sp_ichol_coo()
