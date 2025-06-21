function [X, relres] = mumps_solver(K, F)

id = initmumps;
id.SYM = 1;

% here JOB = -1, the call to MUMPS will initialize C 
% and fortran MUMPS structure
id = dmumps(id);

id.JOB = 6;
%%%%%%% BEGIN OPTIONAL PART TO ILLUSTRATE THE USE OF MAXIMUM TRANSVERSAL
id.ICNTL(7) = 5;
id.ICNTL(6) = 1;
id.ICNTL(8) = 7;
id.ICNTL(14) = 80;
% we set the rigth hand side
id.RHS = F;
%call to mumps
id = dmumps(id, K);
% we activate the numerical maximun transversal 
fprintf('total number of nonzeros in factors %d\n', id.INFOG(10));

%%%%%%% END OPTIONAL PART %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = id.SOL;
relres = norm((K*X - F)/norm(F));
% destroy mumps instance
id.JOB = -2;
id = dmumps(id);
