import numpy as np
from .timeline import ChebyshevTimeLine


class TimeIntegratorAlgorithm():
    def __init__(self):

    def advance(self):
        dt = self.timeline.get_current_time_step_length()
        self.solve(dt)

    def get_current_linear_system(self):
        pass

    def run(self, timeline, uh):
        while ~self.timeline.stop():
            self.advance()

class SDCTimeIntegratorAlgorithm():

    def __init__(self, nupdate=10):
        self.nupdate = nupdate

    def run(self, ctimeline, uh):
        for i in range(self.nupdate):
            while ~ctimeline.stop():
                dt = ctimeline.get_current_time_step_length()
                self.solve(dt)
            self.update()

    def update(self):
        pass




