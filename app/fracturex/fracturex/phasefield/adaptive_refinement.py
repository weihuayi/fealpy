from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.fem.recovery_alg import RecoveryAlg
from fealpy.experimental.mesh.mesh_base import Mesh


class AdaptiveRefinement:
    def __init__(self, mesh, marking_strategy: str = 'recovery',
                 refine_method: str = 'bisect', theta: float = 0.2):
        """
        自适应加密模块初始化。
        
        Parameters:
        ----------
        marking_strategy : str
            元素标记策略，如'recovery', 'residual'等, 默认'recovery'。
        refine_method : str
            网格加密方法，如'nvp', 'bisection', 'red-green'等, 默认'nvp'。
        theta : float
            标记策略的参数，通常控制加密的比例, 默认0.2。
        """
        self.marking_strategy = marking_strategy
        self.refine_method = refine_method
        self.theta = theta
        self.mesh = mesh
        
    def perform_refinement(self, solution):
        """
        执行自适应网格加密。
        
        Parameters:
        ----------
        mesh : object
            当前的网格对象。
        solution : object
            当前解（相场、位移等）。
        error_estimator : object
            误差估计器对象，用于计算每个单元的误差。
        
        Returns:
        -------
        new_mesh : object
            更新后的网格。
        """
        
        marked_cells = self.mark_cells(solution)
        new_mesh = self.refine_mesh(marked_cells)

        return new_mesh

    def mark_cells(self, solution):
        """
        根据误差指示器和标记策略选择需要加密的单元。
        
        Parameters:
        ----------
        mesh : object
            当前网格。
        soultion : array
            每个单元的误差值。
        
        Returns:
        -------
        marked_cells : array
            被标记为需要加密的单元索引。
        """
        mesh = self.mesh
        if self.marking_strategy == 'recovery':
            # 使用恢复型误差估计进行标记
            error_indicator = RecoveryAlg.recovery_estimate(mesh, solution)

            # 2. 标记需要加密的单元
            marked_cells = Mesh.mark(error_indicator)
        elif self.marking_strategy == 'residual':
            # 使用残差基误差估计进行标记
            # 实现残差基误差估计标记逻辑
            pass
        else:
            raise ValueError(f"未知的标记策略: {self.marking_strategy}")

        return marked_cells

    def refine_mesh(self, isMarkedCell):
        """
        根据标记的单元执行网格加密。
        
        Parameters:
        ----------
        mesh : object
            当前的网格。
        marked_cells : array
            被标记为需要加密的单元索引。
        
        Returns:
        -------
        new_mesh : object
            加密后的新网格。
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        '''
        cm = mesh.entity_measure('cell')
        if GD == 3:
            hmin = model.l0**3/200
        else:
            hmin = (model.l0/4)**2
        '''
        #isMarkedCell = bm.logical_and(isMarkedCell, cm > hmin)
        if bm.any(isMarkedCell):
            if GD == 2:
                if self.refine_method == 'bisect':
                    self.bisect_refine_2d(isMarkedCell)
                elif self.refine_method == 'red-green':
                    self.redgreen_refine_2d(isMarkedCell)
            elif GD == 3:
                self.bisect_refine_3d(isMarkedCell)
            else:
                raise ValueError(f"未知的加密方法: {self.refine_method}")
