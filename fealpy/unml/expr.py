
class Expr:
    def __add__(self, other):
        return Add(self, other)
    def __mul__(self, other):
        return Mul(self, other)
    def __rmul__(self, other):
        return Mul(other, self)
    def __str__(self):
        return self.__class__.__name__


# Arithmetic nodes
class Add(Expr):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __str__(self):
        return f"({self.a} + {self.b})"

class Mul(Expr):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __str__(self):
        return f"({self.a} * {self.b})"

class Grad(Expr):
    def __init__(self, arg):
        self.arg = arg
    def __str__(self):
        return f"grad({self.arg})"

def grad(f):
    return Grad(f)

class Laplace(Expr):
    def __init__(self, arg):
        self.arg = arg
    def __str__(self):
        return f"laplace({self.arg})"

def laplace(f):
    return Laplace(f)

class Dot(Expr):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __str__(self):
        return f"dot({self.a}, {self.b})"

def dot(a, b):
    return Dot(a, b)

class Measure:
    def __init__(self, tag):
        self.tag = tag
    def __rmul__(self, expr):
        return Integral(expr, self)

class Integral:
    def __init__(self, expr, measure):
        self.expr = expr
        self.measure = measure
    def __str__(self):
        return f"integrate({self.expr}) over {self.measure.tag}"


dcell = Measure("dcell")         # finite element volume
dface = Measure("dface")         # face or control surface (FVM, DG)
dnode = Measure("dnode")         # finite difference nodes
dedge = Measure("dedge")         # finite volume edges

dx = dcell
ds = dface
