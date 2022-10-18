import numpy as np


class ODESolver():
    def run(self, x, t, dt, tf):
        """
        @brief 从时刻 t 到 tf 时间积分

        @param[in, out]
        """
        while t < tf:
            self.step(x, t, dt)
