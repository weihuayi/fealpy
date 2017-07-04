import numpy as np

class TimeIntegratorAlgorithm():
    def __init__(self, initTime, stopTime):
        self.currentTime = initTime 
        self.stopTime = stopTime 

    def step(self, dt):
        self.currentTime += dt
        A, b = self.get_current_linear_system()
        X  = self.solve(A, b) 
        return X

    def get_current_time(self):
        return self.currentTime

    def get_stop_time(self):
        return self.stopTime

    def get_step_length(self):
        pass

    def get_current_linear_system(self):
        pass

    def run(self):
        while self.currentTime < self.stopTime: 
            try:
                dt = self.get_step_length()
                currentSolution = self.step(dt)
                self.accept_solution(currentSolution)
            except StopIteration:
                break

    def accept_solution(self, currentSolution):
        pass

    def solve(self, A, b):
        pass


