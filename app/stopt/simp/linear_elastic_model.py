class BoxDomainData2d:
    """
    @brief Dirichlet 边界条件的线弹性问题模型
    @note 本模型假设在二维方形区域 [0,1] x [0,1] 内的线性弹性问题
    """

    def __init__(self, e=1.0, nu=0.3):
        """
        @brief 构造函数
        @param[in] e 弹性模量，默认值为 1.0
        @param[in] nu 泊松比，默认值为 0.3
        """
        self.e = e
        self.nu = nu

        self.lam = self.nu * self.e / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.e / (2 * (1 + self.nu))
