function [V, D] = eigs_mumps(K, M, k, sigma)
[m, n] = size(K);

if nargin == 4
    K = K + sigma*M;
end

% initial the mumps
id = initmumps;
id.SYM = 0; % symetric matrix
id = dmumps(id);

id.JOB = 4; % analysis, factorization
id.ICNTL(7) = 5;
id.ICNTL(6) = 1;
id.ICNTL(8) = 7;
id.ICNTL(14) = 80;
id = dmumps(id, K);

id.JOB = 3;

[V, D] = eigs(@KM, m, k, 'LR');
D = real(diag(D));
[D, I] = sort(D, 1, 'descend');
V = V(:, I);
D = 1./D;

if nargin == 4
    D = D - sigma;
end


% destroy mumps instance
id.JOB = -2;
id = dmumps(id);

function y = KM(x)
    id.RHS = M*x;
    id = dmumps(id, K); % the first solve with analysis and factorization
    y = id.SOL;
end
end
