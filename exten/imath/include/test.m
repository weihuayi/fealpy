A = [1, 0, 0, 0;2, 3, 0, 0;0, 4, 8, 0;5, 0, 0, 8];
A = A + tril(A, -1)';

A = sparse(A);
b = ones(4, 1);

[x0, x1] = tridivide(A, b);
