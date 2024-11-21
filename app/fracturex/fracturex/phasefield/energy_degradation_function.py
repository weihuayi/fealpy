class EnergyDegradationFunction:
    def __init__(self, degradation_type='quadratic', **kwargs):
        """
        Initialize the energy degradation function module.

        Parameters:
        Degradation-type (str): The type of energy degradation function that supports' quadratic ',
        'thrice ',' user_define ', etc.
        Kwargs (dict): Additional parameters are used for different types of degenerate functions, such as exponential factors in exponential functions.
        """
        self.degradation_type = degradation_type
        self.params = kwargs

    def degradation_function(self, d):
        """
        Calculate the energy degradation factor g (d) based on the phase field value d.

        Parameters:
        d (float or numpy array): Phase field value, with a value range of [0,1].

        return:
        g (float or numpy array): Degradation factor g (d).
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
        
    def grad_grad_degradation_function(self, d):
        if self.degradation_type == 'quadratic':
            return self._quadratice_grad_grad_degradation(d)
        elif self.degradation_type == 'user_defined':
            return self._user_defined_grad_grad_degradation(d)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")
        
    def grad_degradation_function_constant_coef(self):
        """
        Get the constant coefficient in the gradient of the energy degradation function.

        Parameters:
        d (float or numpy array): phase field value.

        return:
        c (float or numpy array): The constant coefficient in the gradient of the energy degradation function.
        """
        if self.degradation_type == 'quadratic':
            return -2
        elif self.degradation_type == 'user_defined':
            return self.params.get('constant_coef')
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")


    def _quadratic_degradation(self, d):
        """
        The quadratic energy degradation function g (d)=(1-d) ^ 2.

        Parameters:
        d (float or numpy array): phase field value.

        return:
        g (float or numpy array): Degradation factor g (d).
        """
        eps = 1e-10
        gd = (1 - d)**2 + eps
        return gd
    
    def _quadratic_grad_degradation(self, d):
        """
        The derivative of the quadratic energy degradation function g'(d) = -2(1 - d)。
        """
        g_gd = -2 + 2*d
        return g_gd
    
    def _quadratice_grad_grad_degradation(self, d):
        """
        The second derivative of the quadratic energy degradation function g''(d) = 2。
        """
        return 2

    def _thrice_degradation(self, d):
        """
        The thrice degradation function g(d) = 3(1-d)^2-2(1-d)^3。

        Parameters:
        d (float or numpy array): phase field value.

        return:
        g (float or numpy array): Degradation factor g (d).
        """
        eps = 1e-10
        gd = 3*(1 - d)**2 - 2*(1-d)**3 + eps
        return gd

    def _user_defined_degradation(self, d):
        """
        User defined energy degradation function, which can be in any user-defined form.

        Parameters:
        d (float or numpy array): phase field value.

        return:
        g (float or numpy array): Degradation factor g (d).
        """
        custom_function = self.params.get('custom_function')
        if custom_function is None:
            raise ValueError("For user_defined degradation, 'custom_function' must be provided.")
        return custom_function(d)
    
    def _user_defined_grad_degradation(self, d):
        """
        The derivative of the user-defined energy degradation function.
        """
        custom_grad_function = self.params.get('custom_grad_function')
        if custom_grad_function is None:
            raise ValueError("For user_defined degradation, 'custom_grad_function' must be provided.")
        return custom_grad_function(d)
    
    def _user_defined_grad_grad_degradation(self, d):
        """
        The second derivative of the user-defined energy degradation function.
        """
        custom_grad_grad_function = self.params.get('custom_grad_grad_function')
        if custom_grad_grad_function is None:
            raise ValueError("For user_defined degradation, 'custom_grad_grad_function' must be provided.")
        return custom_grad_grad_function(d)

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

