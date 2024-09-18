from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.fem.recovery_alg import RecoveryAlg
from fealpy.experimental.mesh.mesh_base import Mesh


class AdaptiveRefinement:
    def __init__(self, mesh, marking_strategy: str = 'recovery',
                 refine_method: str = 'nvp', theta: float = 0.2):
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
        
        return new_mesh


    def bisect_refine_3d(self, isMarkedCell):
        """
        @brief 四面体二分法加密策略
        """
        NQ = self.H.shape[0] if len(self.H) > 1 else 1
        data = {'nodedata':[self.uh[:, 0], self.uh[:, 1], self.uh[:, 2],
            self.d]}
        for i in range(NQ):
            data = {'celldata':[self.H[i, :]]}

        option = self.mesh.bisect_options(data=data, disp=False, HB=True)
        self.mesh.bisect(isMarkedCell, options=option)
        print('mesh refine')      
       
        # 更新加密后的空间
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        NC = self.mesh.number_of_cells()
        self.uh = self.space.function(dim=self.GD)
        self.d = self.space.function()
        self.H = np.zeros((NQ, NC), dtype=np.float64)  # 分片常数

        self.uh[:, 0] = option['data']['nodedata'][0]
        self.uh[:, 1] = option['data']['nodedata'][1]
        self.uh[:, 2] = option['data']['nodedata'][2]
        self.d[:] = option['data']['nodedata'][3]
        for i in range(NQ):
            self.H[i, :] = option['data']['celldata'][i]

       
    def bisect_refine_2d(self, isMarkedCell):
        """
        @brief 二分法加密策略
        """
        NQ = self.H.shape[0] if len(self.H) > 1 else 1
        
        if self.p == 1:
            data = {'uh0': self.uh[:, 0], 'uh1': self.uh[:, 1], 'd': self.d,}
        else:
            cell2dof = self.space.cell_to_dof()
            data = {'uh0': self.uh[cell2dof, 0], 'uh1': self.uh[cell2dof, 1],
                    'd': self.d[cell2dof],}

        for i in range(NQ):
            data[f'H_{i}'] = self.H[i, :]
        option = self.mesh.bisect_options(data=data, disp=False)
        self.mesh.bisect(isMarkedCell, options=option)
        print('mesh refine')      
       
        # 更新加密后的空间
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        NC = self.mesh.number_of_cells()
        self.uh = self.space.function(dim=self.GD)
        self.d = self.space.function()
        self.H = np.zeros((NQ, NC), dtype=np.float64)  # 分片常数
        
        if self.p == 1 :
            self.uh[:, 0] = option['data']['uh0']
            self.uh[:, 1] = option['data']['uh1']
            self.d[:] = option['data']['d']
        else:
            cell2dof = self.space.cell_to_dof()
            self.uh[cell2dof.reshape(-1), 0] = option['data']['uh0'].reshape(-1)
            self.uh[cell2dof.reshape(-1), 1] = option['data']['uh1'].reshape(-1)
            self.d[cell2dof.reshape(-1)] = option['data']['d'].reshape(-1)

        for i in range(NQ):
            self.H[i, :] = option['data'][f'H_{i}']

       
    def redgreen_refine_2d(self, isMarkedCell):
        """
        @brief 红绿加密策略
        """
        NQ = self.H.shape[0] if len(self.H) > 1 else 1
        for i in range(NQ):
            Hi = self.H[i, :]
            self.mesh.celldata[f'H_{i}'] = Hi
#        self.mesh.celldata['H'] = self.H
        mesho = copy.deepcopy(self.mesh)

        spaceo = LagrangeFESpace(mesho, p=self.p, doforder='vdims')
        uh0 = spaceo.function()
        uh1 = spaceo.function()
        d0 = spaceo.function()
        uh0[:] = self.uh[:, 0]
        uh1[:] = self.uh[:, 1]
        d0[:] = self.d[:]

        self.mesh.refine_triangle_rg(isMarkedCell)
        print('mesh refine')      
       
        # 更新加密后的空间
        self.space = LagrangeFESpace(self.mesh, p=self.p, doforder='vdims')
        NC = self.mesh.number_of_cells()
        self.uh = self.space.function(dim=self.GD)
        self.d = self.space.function()
        self.H = np.zeros((NQ, NC), dtype=np.float64)  # 分片常数

        
        self.uh[:, 0] = self.space.interpolation_fe_function(uh0)
        self.uh[:, 1] = self.space.interpolation_fe_function(uh1)
        
        self.d[:] = self.space.interpolation_fe_function(d0)
        
        for i in range(NQ):
            Hi = np.zeros(NC, dtype=np.float64)
            self.mesh.interpolation_cell_data(mesho, datakey=[f'H_{i}'])
            self.H[i, :] = Hi
#        self.mesh.interpolation_cell_data(mesho, datakey=['H'])
        print('interpolation cell data:', NC)      
