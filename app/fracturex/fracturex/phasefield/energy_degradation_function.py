class EnergyDegradationFunction:
    def __init__(self, degradation_type='quadratic', **kwargs):
        """
        初始化能量退化函数模块。

        参数:
        degradation_type (str): 能量退化函数的类型，支持 'quadratic',
        'thrice', 'user_defined' 等。
        kwargs (dict): 额外的参数用于不同类型的退化函数，例如指数函数中的指数因子。
        """
        self.degradation_type = degradation_type
        self.params = kwargs

    def degradation_function(self, d):
        """
        根据相场值 d 计算能量退化因子 g(d)。

        参数:
        d (float or numpy array): 相场值，取值范围为 [0, 1]。

        返回:
        g (float or numpy array): 退化因子 g(d)。
        """
        if self.degradation_type == 'quadratic':
            return self._quadratic_degradation(d)
        elif self.degradation_type == 'thrice':
            return self._thrice_degradation(d)
        elif self.degradation_type == 'user_defined':
            return self._user_defined_degradation(d)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")

    def grad_degradation_function(self, d):
        if self.degradation_type == 'quadratic':
            return self._quadratic_grad_degradation(d)
        elif self.degradation_type == 'user_defined':
            return self._user_defined_grad_degradation(d)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")


    def _quadratic_degradation(self, d):
        """
        二次能量退化函数 g(d) = (1 - d)^2。

        参数:
        d (float or numpy array): 相场值。

        返回:
        g (float or numpy array): 退化因子 g(d)。
        """
        eps = 1e-10
        gd = (1 - d)**2 + eps
        return gd
    
    def _quadratic_grad_degradation(self, d):
        """
        二次能量退化函数的导数 g'(d) = -2(1 - d)。
        """
        eps = 1e-10
        gd = -2*(1 - d) + eps
        return gd

    def _thrice_degradation(self, d):
        """
        指数型能量退化函数 g(d) = 3(1-d)^2-2(1-d)^3。

        参数:
        d (float or numpy array): 相场值。
        alpha (float): 指数函数中的指数因子，通常为一个正数。

        返回:
        g (float or numpy array): 退化因子 g(d)。
        """
        eps = 1e-10
        gd = 3*(1 - d)**2 - 2*(1-d)**3 + eps
        return gd

    def _user_defined_degradation(self, d):
        """
        用户自定义能量退化函数，可以是任何用户定义的形式。

        参数:
        d (float or numpy array): 相场值。

        返回:
        g (float or numpy array): 退化因子 g(d)。
        """
        custom_function = self.params.get('custom_function')
        if custom_function is None:
            raise ValueError("For user_defined degradation, 'custom_function' must be provided.")
        return custom_function(d)

    def plot_degradation_function(self, d_values):
        """
        绘制能量退化函数的曲线。

        参数:
        d_values (numpy array): 一组相场值 d 用于绘制函数曲线。
        """
        import matplotlib.pyplot as plt

        g_values = self.degradation_function(d_values)
        plt.plot(d_values, g_values, label=f'{self.degradation_type} degradation')
        plt.xlabel('Phase field variable d')
        plt.ylabel('Degradation function g(d)')
        plt.title('Energy Degradation Function')
        plt.legend()
        plt.grid(True)
        plt.show()

