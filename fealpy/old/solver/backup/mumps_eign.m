function [V, D] = mumps_eign(K, M, k)

[m, n] = size(K);

% initial the mumps
id = initmumps;
id.SYM = 1; % symetric matrix
id = dmumps(id);

id.JOB = 6; % analysis, factorization and solve
id.ICNTL(7) = 5;
id.ICNTL(6) = 1;
id.ICNTL(8) = 7;
id.ICNTL(14) = 80;

kk = max(k, 20); 

Q = zeros(n, kk+1);
Q(:, 1) = rand(n ,1);
Q(:, 1) = Q(:, 1)/norm(Q(:, 1)); % 

a = zeros(kk+1, 1);
b = zeros(kk, 1);

id.RHS = M*Q(:, 1);
id = dmumps(id, K); % the first solve with analysis and factorization
u = id.SOL;

id.JOB = 3; % just solve 

a(1) = sum(Q(:, 1).*u);
u = u - a(1)*Q(:, 1);
b(1) = norm(u);
Q(:, 2) = u/b(1);
for i=2:kk
    %solve
    id.RHS = M*Q(:, i);
    id = dmumps(id, K);
    u = id.SOL;
    a(i) = sum(Q(:, i).*u); 
    u = u - a(i)*Q(:, i) - b(i-1)*Q(:, i-1);
    b(i) = norm(u);
    Q(:, i+1) = u/b(i);
end
T = diag(a) + diag(b, -1) + diag(b, 1);
[V, D] = eig(T); 

D = diag(D);
[D, I] = sort(D, 1, 'descend');
V = V(:, I);

V = Q*V;
V = V(:, 1:k);
D = 1./D(1:k);

% destroy mumps instance
id.JOB = -2;
id = dmumps(id);
end
