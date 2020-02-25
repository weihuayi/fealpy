# https://github.com/bamos/block
# An intelligent block matrix library for numpy, PyTorch, and beyond. Crafted 
# by Brandon Amos with significant contributions by Eric Wong.

import numpy as np
import scipy.sparse.linalg as sla
import scipy.sparse as sp

try:
    import torch
    from torch.autograd import Variable
except:
    pass


import re
from abc import ABCMeta, abstractmethod


def block(rows, dtype=None, arrtype=None):
    if (not _is_list_or_tup(rows)) or len(rows) == 0 or \
       np.any([not _is_list_or_tup(row) for row in rows]):
        raise RuntimeError('''
Unexpected input: Expected a non-empty list of lists.
If you are interested in helping expand the functionality
for your use case please send in an issue or PR at
http://github.com/bamos/block''')

    rowLens = [len(row) for row in rows]
    if len(np.unique(rowLens)) > 1:
        raise RuntimeError('''
Unexpected input: Rows are not the same length.
Row lengths: {}'''.format(rowLens))

    nRows = len(rows)
    nCols = rowLens[0]
    rowSizes = np.zeros(nRows, dtype=int)
    colSizes = np.zeros(nCols, dtype=int)

    backend = _get_backend(rows, dtype, arrtype)

    for i, row in enumerate(rows):
        for j, elem in enumerate(row):
            if backend.is_complete(elem):
                rowSz, colSz = backend.extract_shape(elem)
                rowSizes[i] = rowSz
                colSizes[j] = colSz
            elif hasattr(elem, 'shape'):
                rowSz, colSz = elem.shape
                rowSizes[i] = rowSz
                colSizes[j] = colSz
            elif hasattr(elem, 'size'):
                rowSz, colSz = elem.size()
                rowSizes[i] = rowSz
                colSizes[j] = colSz

    cRows = []
    for row, rowSz in zip(rows, rowSizes):
        rowSz = int(rowSz)
        if rowSz == 0:
            continue
        cCol = []
        for elem, colSz in zip(row, colSizes):
            colSz = int(colSz)
            if colSz == 0:
                continue
            # TODO: Check types.
            if backend.is_complete(elem):
                cElem = elem
            elif isinstance(elem, float) or isinstance(elem, int):
                cElem = backend.build_full((rowSz, colSz), elem)
            elif isinstance(elem, str):
                if elem == 'I':
                    assert(rowSz == colSz)
                    cElem = backend.build_eye(rowSz)
                elif elem == '-I':
                    assert(rowSz == colSz)
                    cElem = -backend.build_eye(rowSz)
                else:
                    assert(False)
            else:
                cElem = backend.convert(elem)
            cCol.append(cElem)
        cRows.append(cCol)

    return backend.build(cRows)


def block_diag(elems, dtype=None, arrtype=None):
    n = len(elems)
    return block([[0] * i + [elem] + [0] * (n - 1 - i)
                  for i, elem in enumerate(elems)],
                 dtype=dtype, arrtype=arrtype)


def block_tridiag(main, upper, lower):
    n = len(main)
    assert len(main) == len(upper) + 1
    assert len(main) == len(lower) + 1
    mat = ()
    for i in range(n):
        tup = ()
        for j in range(n):
            if (i==j):   tup = (*tup, main[i])
            elif (i==j-1): tup = (*tup, upper[-i])
            elif (i==j+1): tup = (*tup, lower[i-1])
            else: tup = (*tup,0)
        mat = (*mat,tup)
    return block(mat)

def _is_list_or_tup(x):
    return isinstance(x, list) or isinstance(x, tuple)


def _get_backend(rows, dtype, arrtype):
    if arrtype == np.ndarray and dtype is not None:
        return NumpyBackend(arrtype, dtype)
    elif arrtype == sla.LinearOperator:
        return LinearOperatorBackend(dtype)
    elif arrtype is not None and re.search('torch\..*Tensor', repr(arrtype)):
        return TorchBackend(dtype)
    elif arrtype is not None and re.search('torch\..*(Variable|Parameter)', repr(arrtype)):
        return TorchVariableBackend(dtype)
    else:
        npb = NumpyBackend()
        tb = TorchBackend()
        lob = LinearOperatorBackend()
        tvb = TorchVariableBackend()
        for row in rows:
            for elem in row:
                if npb.is_complete(elem) and elem.size > 0:
                    if dtype is None:
                        dtype = type(elem[0, 0])
                    if arrtype is None:
                        arrtype = type(elem)
                    return NumpyBackend(dtype, arrtype)
                elif tb.is_complete(elem):
                    return TorchBackend(type(elem))
                elif lob.is_complete(elem):
                    return LinearOperatorBackend(elem.dtype)
                elif tvb.is_complete(elem):
                    return TorchVariableBackend(type(elem.data))

    assert(False)


class Backend():
    __metaclass__ = ABCMeta

    @abstractmethod
    def extract_shape(self, x): pass

    @abstractmethod
    def build_eye(self, n): pass

    @abstractmethod
    def build_full(self, shape, fill_val): pass

    @abstractmethod
    def convert(self, x): pass

    @abstractmethod
    def build(self, rows): pass

    @abstractmethod
    def is_complete(self, rows): pass


class NumpyBackend(Backend):

    def __init__(self, dtype=None, arrtype=None):
        self.dtype = dtype
        self.arrtype = arrtype

    def extract_shape(self, x):
        return x.shape

    def build_eye(self, n):
        return np.eye(n)

    def build_full(self, shape, fill_val):
        return np.full(shape, fill_val, self.dtype)

    def convert(self, x):
        assert(False)

    def build(self, rows):
        return np.bmat(rows)

    def is_complete(self, x):
        return isinstance(x, np.ndarray)


class TorchBackend(Backend):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def extract_shape(self, x):
        return x.size()

    def build_eye(self, n):
        return torch.eye(n).type(self.dtype)

    def build_full(self, shape, fill_val):
        return fill_val * torch.ones(*shape).type(self.dtype)

    def convert(self, x):
        assert(False)

    def build(self, rows):
        compRows = []
        for row in rows:
            compRows.append(torch.cat(row, 1))
        return torch.cat(compRows)

    def is_complete(self, x):
        return (re.search('torch\..*Tensor', str(x.__class__)) is not None) \
            and x.ndimension() == 2


class TorchVariableBackend(TorchBackend):
    def build_eye(self, n):
        return Variable(super().build_eye(n))

    def build_full(self, shape, fill_val):
        return Variable(super().build_full(shape, fill_val))

    def convert(self, x):
        if TorchBackend.is_complete(self, x):
            return Variable(x)
        assert(False)

    def is_complete(self, x):
        return re.search('torch\..*(Variable|Parameter)', str(x.__class__))


class LinearOperatorBackend(Backend):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def extract_shape(self, x):
        return x.shape

    def build_eye(self, n):
        def identity(v): return v
        return sla.LinearOperator(shape=(n, n),
                                  matvec=identity,
                                  rmatvec=identity,
                                  matmat=identity,
                                  dtype=self.dtype)

    def build_full(self, shape, fill_val):
        m, n = shape
        if fill_val == 0:
            return shape
        else:
            def matvec(v):
                return v.sum() * fill_val * np.ones(m)

            def rmatvec(v):
                return v.sum() * fill_val * np.ones(n)

            def matmat(M):
                return M.sum(axis=0) * fill_val * np.ones((m, M.shape[1]))

            return sla.LinearOperator(shape=shape,
                                      matvec=matvec,
                                      rmatvec=rmatvec,
                                      matmat=matmat,
                                      dtype=self.dtype)

    def convert(self, x):
        if (isinstance(x, (np.ndarray, sp.spmatrix))):
            return sla.aslinearoperator(x)
        else:
            assert(False)

    def build(self, rows):
        col_sizes = [lo.shape[1] if self.is_complete(lo) else lo[1]
                     for lo in rows[0]]
        col_idxs = np.cumsum([0] + col_sizes)
        row_sizes = [row[0].shape[0] if self.is_complete(row[0]) else row[0][0]
                     for row in rows]
        row_idxs = np.cumsum([0] + row_sizes)
        m, n = sum(row_sizes), sum(col_sizes)

        def matvec(v):
            out = np.zeros(m)
            for row, i, j in zip(rows, row_idxs[:-1], row_idxs[1:]):
                out[i:j] = sum(lo.matvec(v[k:l]) for lo, k, l in
                               zip(row, col_idxs[:-1], col_idxs[1:])
                               if self.is_complete(lo))
            return out

        # The transposed list
        cols = zip(*rows)

        def rmatvec(v):
            out = np.zeros(n)
            for col, i, j in zip(cols, col_idxs[:-1], col_idxs[1:]):
                out[i:j] = sum(lo.rmatvec(v[k:l]) for lo, k, l in
                               zip(col, row_idxs[:-1], row_idxs[1:])
                               if self.is_complete(lo))
            return out

        def matmat(M):
            out = np.zeros((m, M.shape[1]))
            for row, i, j in zip(rows, row_idxs[:-1], row_idxs[1:]):
                out[i:j] = sum(lo.matmat(M[k:l]) for lo, k, l in
                               zip(row, col_idxs[:-1], col_idxs[1:])
                               if self.is_complete(lo))
            return out

        return sla.LinearOperator(shape=(m, n),
                                  matvec=matvec,
                                  rmatvec=rmatvec,
                                  matmat=matmat,
                                  dtype=self.dtype)

    def is_complete(self, x):
        return isinstance(x, sla.LinearOperator)
