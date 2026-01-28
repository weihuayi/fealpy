from typing import Iterable, List, Tuple, Union, Dict
from fealpy.functionspace import TensorFunctionSpace
from scipy.sparse import coo_matrix, csr_matrix
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.backend import TensorLike
from fealpy.backend import bm

ArrayLike = Union[list, Tuple,TensorLike]

class PeriodicBC:
    """
    周期边界约束：
    - 给定 master(target) 与 slave(source) 的自由度配对
    - 构造识别/压缩算子 P ∈ R^{m×n}，将周期等价类聚合并删除重复自由度
      A' = P A P^T, b' = P b, u_full = P^T u_reduced
    """
    def __init__(self, space, pairs: List[Tuple[int, int]]):
        self.space = space
        self.ndofs = space.number_of_global_dofs()
        # master -> [slaves]
        self.master_to_slaves: Dict[int, List[int]] = {}
        self.slaves_set = set()
        for m, s in pairs:
            m, s = int(m), int(s)
            if m == s:
                continue
            self.master_to_slaves.setdefault(m, []).append(s)
            self.slaves_set.add(s)

        # 保留的行（代表自由度）：去除所有 slave
        all_idx = bm.arange(self.ndofs)
        if len(self.slaves_set) > 0:
            slaves = bm.array(sorted(self.slaves_set, key=int), dtype=bm.int32)
        else:
            slaves = bm.array([], dtype=bm.int32)
            
        self.keep_indices = bm.setdiff1d(all_idx, slaves)

        # old_dof -> new_row 映射（仅 keep 的行）
        self._row_map = {int(old): int(new) for new, old in enumerate(self.keep_indices)}

        # 构造 P
        self.P = self._build_P()

    @staticmethod
    def dof_pair(space, master_face_idx,slave_face_idx,no_periodic_face_idx):
        """
        给出周期边界边的编号,与非周期边界边的编号
        注意：master_edge_idx 与 slave_edge_idx 必须等长
        且角点处会优先被划分到非周期边界边中，以避免重复
        
        Parameters:
            master_edge_idx: TensorLike
            slave_edge_idx: TensorLike
            no_periodic_edge_idx: TensorLike
        Returns:
            mdof: TensorLike, master dof
            sdof: TensorLike, slave dof
            nodof: TensorLike, no periodic dof
        """
        if isinstance(space, TensorFunctionSpace):
            print("Using scalar space for periodic BC")
            scalar_space = space.scalar_space
        else:
            scalar_space = space
            
        e2dof = scalar_space.edge_to_dof()
        intpoint = scalar_space.interpolation_points()
        
        mdof = e2dof[master_face_idx, :].flatten()
        sdof = e2dof[slave_face_idx, :].flatten()
        nodof = e2dof[no_periodic_face_idx, :].flatten()
        # 对齐与排序
        mdof = bm.setdiff1d(mdof,nodof)
        sdof = bm.setdiff1d(sdof,nodof)
        iy_m = intpoint[mdof, 1]
        iy_s = intpoint[sdof, 1]
        sm = bm.argsort(iy_m)
        ss = bm.argsort(iy_s)
        mdof = mdof[sm]
        sdof = sdof[ss]
        
        nodof = bm.unique(nodof)
        if isinstance(space, TensorFunctionSpace):
            dnim = bm.max(space.dof_shape)
            scalar_ndof = scalar_space.number_of_global_dofs()
            shifts = bm.arange(dnim) * scalar_ndof
            mdof = bm.concat([mdof + shift for shift in shifts])
            sdof = bm.concat([sdof + shift for shift in shifts])
            nodof = bm.concat([nodof + shift for shift in shifts])

        return mdof, sdof, nodof
        
    @classmethod
    def from_pairs(cls, space, master_indices: ArrayLike, slave_indices: ArrayLike):
        """
        通过等长数组构造：
        master_indices[i] 与 slave_indices[i] 为同一周期等价类，master 为代表(master)，slave 为被删除(slave)。
        """
        ti = bm.array(master_indices, dtype=bm.int32).tolist()
        si = bm.array(slave_indices, dtype=bm.int32).tolist()
        assert len(ti) == len(si), "master_indices 与 slave_indices 长度必须一致"
        pairs = list(zip(ti, si))
        return cls(space=space, pairs=pairs)
    
    @classmethod
    def from_bdface(cls, space,master_bdface_idx: ArrayLike, 
                        slave_bdface_idx: ArrayLike,
                        no_periodic_bdface_idx: ArrayLike):
        """
        从边界面索引构造周期边界条件
        通过边界面索引确定自由度配对关系
        """
        mdof, sdof,nodof = cls.dof_pair(space, master_bdface_idx, 
                                     slave_bdface_idx, 
                                     no_periodic_bdface_idx)
        pairs = list(zip(mdof.tolist(), sdof.tolist()))
        obj = cls(space=space, pairs=pairs)
        # 实例属性，避免不同实例间互相覆盖
        obj.mdof = mdof
        obj.sdof = sdof
        obj.nodof = nodof
        return obj
        
    def _build_P(self) -> COOTensor:
        """
        构造稀疏 COOTensor P ∈ R^{mxn}:
        - 对于每个保留自由度 i,P[row_map[i], i] = 1
        - 对于每个 slave s 属于 master m,P[row_map[m], s] = 1
        """
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        # 每个保留自由度的单位项
        for i in self.keep_indices.tolist():
            r = self._row_map[int(i)]
            rows.append(r)
            cols.append(int(i))
            data.append(1.0)

        # 将每个 slave 聚合进其 master 的行
        for m, slaves in self.master_to_slaves.items():
            if m not in self._row_map:
                continue
            r = self._row_map[m]
            for s in slaves:
                rows.append(r)
                cols.append(int(s))
                data.append(1.0)

        indices = bm.stack([bm.array(rows, dtype=bm.int32), bm.array(cols, dtype=bm.int32)])  # (2, nnz)
        values = bm.array(data, dtype=bm.float64)
        spshape = (len(self.keep_indices), self.ndofs)
        return COOTensor(indices, values, spshape=spshape).tocsr()

    def apply(self, A: Union[COOTensor, CSRTensor, coo_matrix, csr_matrix],
              *vecs: TensorLike):
        """
        应用周期边界到矩阵和任意数量的向量：
        返回 (A_reduced, *vecs_reduced)
        """
        A_reduced = self.apply_to_matrix(A)
        vecs_reduced = self.apply_to_vector(*vecs)
        return (A_reduced, *vecs_reduced)
    
    def apply_to_vector(self, *vecs: TensorLike):
        """
        应用周期边界到任意数量的向量：
        返回 (*vecs_reduced)
        """
        P = self.P
        vecs_reduced = []
        for v in vecs:
            v_reduced = P @ v
            vecs_reduced.append(v_reduced)

        return (*vecs_reduced,)
    
    def apply_to_matrix(self, A: Union[COOTensor, CSRTensor, coo_matrix, csr_matrix]):
        """
        应用周期边界到矩阵：
        返回 A_reduced
        """
        A_csr = self._as_csr_tensor(A)
        P = self.P
        A_reduced = P @ A_csr @ P.T
        return A_reduced

    def lift(self, u_reduced):
        """
        将解从压缩空间还原到完整自由度：u_full = P^T u_reduced
        """
        return (self.P.T @ u_reduced)

    def _as_csr_tensor(self, M: Union[COOTensor, CSRTensor, coo_matrix, csr_matrix]) -> CSRTensor:
        if isinstance(M, CSRTensor):
            return M
        if isinstance(M, COOTensor):
            return M.tocsr()
        if isinstance(M, (coo_matrix, csr_matrix)):
            coo = M.tocoo()
            indices = bm.stack([bm.array(coo.row, dtype=bm.int32), bm.array(coo.col, dtype=bm.int32)])
            values = bm.array(coo.data, dtype=bm.float64)
            coo_t = COOTensor(indices, values, spshape=coo.shape)
            return coo_t.tocsr()
        raise TypeError("不支持的矩阵类型")