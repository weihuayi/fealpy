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
    def assembly(self, *, retain_ints: bool=False) -> TensorLike: ...
    @overload
    def assembly(self, *, format: Literal['coo'], retain_ints: bool=False) -> COOTensor: ...
    @overload
    def assembly(self, *, format: Literal['dense'], retain_ints: bool=False) -> TensorLike: ...
    def assembly(self, *, format='dense', retain_ints: bool=False): 
        V = [i.assembly(format=format, retain_ints=retain_ints)for i in self.blocks]
        self._V = bm.concatenate(V)
        return self._V

Form.register(LinearBlockForm)

