from typing import List

from .. import logger
from ..typing import Size, TensorLike
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor
from .form import Form
from ..sparse import COOTensor, CSRTensor

class BlockForm(Form):
    #为什么有这一行
    _M = None

    def __init__(self, blocks:List):
        self.blocks = blocks
        self.sparse_shape = self._get_sparse_shape() 
         
    def _get_sparse_shape(self):
         ## 检查能否拼接
         ## batch 情况
         block_shape = bm.array([[block.sparse_shape if block is not None else (0,0) for block in row] for row in self.blocks], dtype=bm.int32)
         self.block_shape = block_shape
         self.nrows = block_shape.shape[0]
         self.ncols = block_shape.shape[1]
         shape0 = bm.sum(bm.max(block_shape[..., 0], axis=1))
         shape1 = bm.sum(bm.max(block_shape[..., 1], axis=0))
         return (shape0, shape1)

    @property
    def shape(self) -> Size:
        return self.sparse_shape

    def assembly(self, format='coo'):
        a = bm.max(self.block_shape[...,0], axis=1)
        row_offset = bm.cumsum(bm.max(self.block_shape[...,0],axis = 1), axis=0)
        col_offset = bm.cumsum(bm.max(self.block_shape[...,1],axis = 0), axis=0)
        row_offset = bm.concatenate((bm.array([0]),row_offset))
        col_offset = bm.concatenate((bm.array([0]),col_offset))
        
        # type类型处理
        indices = bm.empty((2, 0), dtype=bm.int32)
        values = bm.empty((0,), dtype=bm.float64)
        sparse_shape = self.shape
        
        for i in range(self.nrows):
            for j in range(self.ncols):
                block = self.blocks[i][j]
                if block is None:
                    continue
                ## 不用assemble的方法
                block_matrix = block.assembly()
                block_indices = block_matrix.indices() + bm.array([[row_offset[i]], [col_offset[j]]])
                block_values = block_matrix.values() 
                indices = bm.concatenate((indices, block_indices), axis=1)
                values = bm.concatenate((values, block_values))
        M = COOTensor(indices, values, sparse_shape) 
        if format == 'csr':
            self._M = M.coalesce().tocsr()
        elif format == 'coo':
            self._M = M.coalesce()
        else:
            raise ValueError(f"Unknown format {format}.")
        logger.info(f"Block form matrix constructed, with shape {list(self._M.shape)}.")
        return self._M
    
    
    def __matmul__(self, u: TensorLike):
        if self._M is not None:
            return self._M @ u
        
        # u的不同ndim情况
        kwargs = bm.context(u)
        v = bm.zeros_like(u, **kwargs)

        row_offset = bm.cumsum(bm.max(self.block_shape[...,0],axis = 1))
        col_offset = bm.cumsum(bm.max(self.block_shape[...,1],axis = 0))
        row_offset = bm.concatenate(([0],row_offset))
        col_offset = bm.concatenate(([0],col_offset))
        for i in range(self.nrows):
            for j in range(self.ncols):
                block = self.blocks[i][j]
                if block is None:
                    continue
                v[row_offset[i]:row_offset[i+1]] += block @ u[row_offset[i]:row_offset[i+1]] 
        return v


Form.register(BlockForm)
