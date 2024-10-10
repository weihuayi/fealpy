from typing import Optional, Literal, overload, List

from ..typing import TensorLike
from ..backend import backend_manager as bm
from ..sparse import COOTensor, CSRTensor
from .form import Form
from .integrator import LinearInt

class BlockForm(Form):

    def __init__(self, blocks:List):
        self.blocks = blocks 
        self.sparse_shape = self._get_sparse_shape()
        
    
    def _get_sparse_shape(self):
         shape0 = [bm.sum([block.sparse_shape[0] if block is not None else 0 for block in row]) for row in self.blocks]
         shape1 = [bm.sum([block.sparse_shape[1] if block is not None else 0 for block in row]) for row in self.blocks]
         shape = bm.array([[block.sparse_shape if block is not None else (0,0) for block in row] for row in self.blocks], dtype=bm.int32)
         shapee = [[block.sparse_shape if block is not None else (0,0) for block in row] for row in self.blocks]
         shape0 = bm.sum(shape,axis=2)
         print(shape0)
         print(shape[:,:,0])
         return (shape0, shape1)
         #row_sizes = [max(block.shape[0] if block is not None else 0 for block in row) for row in self.blocks]
         #col_sizes = [max(self.blocks[i][j].shape[1] if self.blocks[i][j] is not None else 0 for i in range(len(self.blocks))) for j in range(len(self.blocks[0]))]
         #print(row_sizesdd) 


    def assemble(self):
        blocks = []
        for i in range(self.nrows):
            row_blocks = []
            for j in range(self.ncols):
                if self.forms[i][j] is None:
                    row_blocks.append(None)
                else:
                    row_blocks.append(self.forms[i][j].assemble())
            blocks.append(row_blocks)
        return blocks

    def __repr__(self):
        return f("BlockForm(shape={self.shape})")
Form.register(BlockForm)
