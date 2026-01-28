class CutGraph:
    """
    [Future Work]
    用于处理非零亏格曲面的切割算法。
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def compute_homology_basis(self):
        """计算同调群基底"""
        raise NotImplementedError("Coming soon in CCG course!")

    def slice_mesh(self, path):
        """沿着路径切开网格"""
        pass