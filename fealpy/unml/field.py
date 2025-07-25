
from .expr import Expr

class Field(Expr):
    def __init__(self, space, name=""):
        self.name = name
        self.space = space
    def __str__(self):
        return self.name or self.__class__.__name__

class TrialFunction(Field):
    def __init__(self, space, name=""):
        super().__init__(space, name)
        self.role = "trial"

class TestFunction(Field):
    def __init__(self, space, name=""):
        super().__init__(space, name)
        self.role = "test"

class Function(Field):
    def __init__(self, space, name=""):
        super().__init__(space, name)
        self.role = "given"
