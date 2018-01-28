import numpy as np

class TimeIntegratorAlgorithm():
    def __init__(self, timeline):
        self.timeline = timeline
        self.current = 0 
        self.stop = len(timeline)

    def step(self):
        self.current += 1 
        A, b = self.get_current_linear_system()
        X  = self.solve(A, b) 
        return X

    def get_current_time(self):
        return self.timeline[self.current]

    def get_stop_time(self):
        return self.timeline[self.stopstopTime] 

    def get_step_length(self):
        return self.timeline[self.current + 1] - self.timeline[self.current]

    def get_current_linear_system(self):
        pass

    def run(self):
        while self.current < self.stop: 
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


