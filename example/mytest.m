e=zeros(4,1)
option.solver = 'CG';
option.tol = 1e-15
option.printlevel = 2
load ls0.mat
[u, info] = amg(A, b, option);
ui = cos(pi*p(:,1)).*cos(pi*p(:,2));
e(1) = sqrt(sum((u - ui).^2)/length(ui))
load ls1.mat
[u, info] = amg(A, b, option);
ui = cos(pi*p(:,1)).*cos(pi*p(:,2));
e(2) = sqrt(sum((u - ui).^2)/length(ui));
load ls2.mat
[u, info] = amg(A, b, option)
ui = cos(pi*p(:,1)).*cos(pi*p(:,2));
e(3) = sqrt(sum((u - ui).^2)/length(ui));
load ls3.mat
[u, info] = amg(A, b, option);
ui = cos(pi*p(:,1)).*cos(pi*p(:,2));
e(4) = sqrt(sum((u - ui).^2)/length(ui));

e(1:end-1)./e(2:end)