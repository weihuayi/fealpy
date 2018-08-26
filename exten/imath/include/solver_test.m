

K = [ 10, 0, 0, 0; 2, 5, 0, 0; 1, 0, 4, 0; 0, 0, 3, 5];
K = K + tril(K, -1)';
K 
F = ones(4, 1)
[I, J, data] = find(K);
I = I - 1;
J = J - 1;
solver(I, J, data, F);
