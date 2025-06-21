from .form import Form
from typing import List,overload, Literal, Optional
from ..typing import TensorLike
from ..sparse import COOTensor

from ..backend import backend_manager as bm 




class LinearBlockForm(Form):
    _V = None

    def __init__(self, blocks:List):
        self.blocks = blocks
        self.sparse_shape = self._get_sparse_shape() 

    def _get_sparse_shape(self):
        shape = [i._get_sparse_shape() for i in self.blocks]
        return (bm.sum(bm.array(shape)), )
    
    @overload
    def assembly(self) -> TensorLike: ...
    @overload
    def assembly(self, *, format: Literal['coo']) -> COOTensor: ...
    @overload
    def assembly(self, *, format: Literal['dense']) -> TensorLike: ...
    def assembly(self, *, format='dense'):
        V = [i.assembly(format=format)for i in self.blocks]
        self._V = bm.concatenate(V)
        return self._V

Form.register(LinearBlockForm)

