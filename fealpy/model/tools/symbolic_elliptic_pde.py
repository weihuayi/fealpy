import sympy as sp
from typing import List, Union, Dict


class SymbolicEllipticPDE:
    """
    General-purpose symbolic toolkit for n-dimensional elliptic PDEs of the form:
    -div(A ∇u) + b·∇u + c u = f
    """

    def __init__(self,
                 u_str: str,
                 A_str: str,
                 b_str: str,
                 c_str: str,
                 var_names: Union[str, List[str]] = "x y"):
        """
        Initialize the symbolic PDE model.

        Parameters:
            u_str (str): Exact solution expression as a string.
            A_str (str): Diffusion coefficient matrix (n×n) as a string.
            b_str (str): Advection vector (n) as a string.
            c_str (str): Reaction coefficient as a string.
            var_names (str or List[str]): Space-separated or list of variable names.

        Returns:
            None: This constructor does not return a value.
        """
        if isinstance(var_names, str):
            var_names = var_names.split()
        self.vars: List[sp.Symbol] = list(sp.symbols(var_names))
        self._locals: Dict[str, sp.Symbol] = dict(zip(var_names, self.vars))

        self.u: sp.Expr   = sp.sympify(u_str, locals=self._locals)
        self.A: sp.Matrix = sp.Matrix(sp.sympify(A_str, locals=self._locals))
        self.b: sp.Matrix = sp.Matrix(sp.sympify(b_str, locals=self._locals))
        self.c: sp.Expr   = sp.sympify(c_str, locals=self._locals)

        self.grad_u: sp.Matrix = None
        self.flux:   sp.Matrix = None
        self.source: sp.Expr   = None

    def compute_gradient(self) -> sp.Matrix:
        """
        Compute the gradient of the solution u.

        Returns:
            sp.Matrix: Gradient vector ∇u.
        """
        self.grad_u = sp.Matrix([sp.diff(self.u, v) for v in self.vars])
        return self.grad_u

    def compute_flux(self) -> sp.Matrix:
        """
        Compute the flux vector q = -A ∇u.

        Returns:
            sp.Matrix: Flux vector q.
        """
        if self.grad_u is None:
            self.compute_gradient()
        self.flux = -self.A * self.grad_u
        return self.flux

    def compute_source(self) -> sp.Expr:
        """
        Compute the source term f = -div(A∇u) + b·∇u + c·u.

        Returns:
            sp.Expr: Source term f.
        """
        n = len(self.vars)
        divAgrad = sum(
            sp.diff(
                sum(self.A[i, j] * sp.diff(self.u, self.vars[j]) for j in range(n)),
                self.vars[i]
            )
            for i in range(n)
        )
        if self.grad_u is None:
            self.compute_gradient()
        adv = self.b.dot(self.grad_u)
        self.source = -divAgrad + adv + self.c * self.u
        return self.source

    def simplify_all(self):
        """
        Simplify all computed expressions (gradient, flux, source).

        Returns:
            None: Simplification is done in-place.
        """
        if self.grad_u is not None:
            self.grad_u = sp.simplify(self.grad_u)
        if self.flux is not None:
            self.flux   = sp.simplify(self.flux)
        if self.source is not None:
            self.source = sp.simplify(self.source)

    def compute_all(self) -> tuple:
        """
        Compute gradient, flux, and source in a single call.

        Returns:
            tuple: A 3-tuple containing (∇u, flux, source).
        """
        self.compute_gradient()
        self.compute_flux()
        self.compute_source()
        return self.grad_u, self.flux, self.source

    def to_python(self, func_name: str = "pde_rhs") -> str:
        """
        Generate Python code for computing gradient, flux, and source.

        Parameters:
            func_name (str): Base name for generated functions.

        Returns:
            str: Generated Python source code.
        """
        self.compute_all()
        from sympy.utilities.codegen import codegen
        [(py_name, py_code), _] = codegen(
            [(func_name + "_grad", self.grad_u),
             (func_name + "_flux", self.flux),
             (func_name + "_source", self.source)],
            language="Python", header=False, empty=False
        )
        return py_code

    def to_c(self, func_name: str = "pde_rhs") -> str:
        """
        Generate C code for computing the source term.

        Parameters:
            func_name (str): Name for the generated C function.

        Returns:
            str: Generated C source code.
        """
        self.compute_all()
        from sympy.utilities.codegen import codegen
        [(c_name, c_code), _] = codegen(
            [(func_name, self.source)],
            language="C", header=False, empty=False
        )
        return c_code


if __name__ == "__main__":
    model = SymbolicEllipticPDE(
        u_str="cos(2*pi*x)*cos(2*pi*y)",
        A_str="[[10,1],[1,10]]",
        b_str="[1, 0.5]",
        c_str="2",
        var_names="x y"
    )
    grad_u, flux, source = model.compute_all()
    model.simplify_all()

    print("∇u =", grad_u)
    print("q  =", flux)
    print("f  =", source)

