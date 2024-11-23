from fealpy.backend import backend_manager as bm


class CrackSurfaceDensityFunction:
    def __init__(self, density_type='AT2', **kwargs):
        self.density_type = density_type
        self.params = kwargs


    def density_function(self, d):
        if self.density_type == 'AT2':
            return self._AT2_density(d)
        elif self.density_type == 'AT1':
            return self._AT1_density(d)
        elif self.density_type == 'user_defined':
            return self._user_defined_density(d)
        else:
            raise ValueError(f"Unknown density type: {self.density_type}")
        
    def grad_density_function(self, d):
        if self.density_type == 'AT2':
            return self._AT2_grad_density(d)
        elif self.density_type == 'AT1':
            return self._AT1_grad_density(d)
        elif self.density_type == 'user_defined':
            return self._user_defined_grad_density(d)
        else:
            raise ValueError(f"Unknown density type: {self.density_type}")
        
    def grad_grad_density_function(self, d):
        if self.density_type == 'AT2':
            return self._AT2_grad_grad_density(d)
        elif self.density_type == 'AT1':
            return self._AT1_grad_grad_density(d)
        elif self.density_type == 'user_defined':
            return self._user_defined_grad_grad_density(d)
        else:
            raise ValueError(f"Unknown density type: {self.density_type}")
        
    def _AT2_density(self, d):
        """
        The AT2 crack surface density function h(d) = d^2, c_d=2.
        """
        return d**2, 2
    
    def _AT2_grad_density(self, d):
        """
        The derivative of the AT2 crack surface density function h'(d) = 2d, c_d=2.
        """
        return 2*d, 2
    
    def _AT2_grad_grad_density(self, d):
        """
        The second derivative of the AT2 crack surface density function h''(d) = 2.
        """
        return 2, 2
    
    def _AT1_density(self, d):
        """
        The AT1 crack surface density function g(d) = d.
        """
        return d, 8/3
    
    def _AT1_grad_density(self, d):
        """
        The derivative of the AT1 crack surface density function g'(d) = 1.
        """
        return 1, 8/3
    
    def _AT1_grad_grad_density(self, d):
        """
        The second derivative of the AT1 crack surface density function g''(d) = 0.
        """
        return 0, 8/3
    
    # User-defined model implementations
    def _user_defined_density(self, d):
        """
        The user-defined crack surface density function h(d).
        """
        if 'density_func' in self.params:
            return self.params['density_func'](d)
        raise NotImplementedError("User-defined density function is not provided.")

    def _user_defined_grad_density(self, d):
        """
        The first derivative of the user-defined crack surface density function h'(d).
        """
        if 'grad_density_func' in self.params:
            return self.params['grad_density_func'](d)
        raise NotImplementedError("User-defined first derivative function is not provided.")

    def _user_defined_grad_grad_density(self, d):
        """
        The second derivative of the user-defined crack surface density function h''(d).
        """
        if 'grad_grad_density_func' in self.params:
            return self.params['grad_grad_density_func'](d)
        raise NotImplementedError("User-defined second derivative function is not provided.")
    