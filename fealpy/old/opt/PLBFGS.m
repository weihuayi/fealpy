classdef PLBFGS < handle
%% 带预条件子的 LBFGS 优化方法
% 
properties
    S % 相邻两步自变量的差
    Y % 相邻两步梯度的差值
    NF % 计算函数值的次数
    L % 不完全 ILU 分解矩阵
    problem % 需要优化的问题
    debug
end
methods
    function obj = PLBFGS(problem)
        obj.problem = problem;
        obj.S = [];
        obj.Y = [];
        obj.NF = 0;
        obj.debug = false;
        P = obj.problem.options.Preconditioner;
        P = sparse(P);
        obj.L = ichol((P+P')/2, struct('type', 'ict','droptol', 1e-5, 'diagcomp', 0.0001));
    end

    function [x, f, g, flag] = run(obj)
    %% 运行算法
        problem = obj.problem;
        options = problem.options;

        x = problem.x0;
        fun = problem.objective;
        lb = problem.lb;
        ub = problem.ub;
        flag = true;
        
        k = 0; % 迭代步

        [f, g] = fun(x);
        gnorm = norm(g);
        pg = g; % 记录上步的梯度
        obj.NF = obj.NF + 1;

        alpha = 1;

        if strcmp(options.Display, 'iter')
            fprintf('The initial status: f=%5.1g, gnorm=%5.1g\n',  f, gnorm)
        end

        flag = 0; % the convergence flag

        for i = 1:options.MaxIterations
            d = - obj.hessian_gradient_prod(g);
            gtd = g'*d;

            if abs(gtd) > 1e4
                fprintf('The norm of the desent direction is too big! Normalize it!\n')
                d = d/norm(d);
                gtd = g'*d;
            end


            if gtd >= 0 | isnan(gtd)
                fprintf('Not descent direction, quit at iteration %d, f = %g, gnorm = %5.1g\n', i, f, gnorm)
            end

            pf = f;
            pg = g;

            [alpha, xalpha, falpha, galpha] = obj.line_search(x, f, g, d, fun, alpha); 
            if alpha > options.StepTolerance
                x = xalpha;
                f = falpha;
                g = galpha;
                gnorm = norm(g);
            else
                if strcmp(options.Display, 'iter')
                    fprintf('The step length alpha %g is smaller than tolerance %g!\n',  alpha, options.StepTolerance);
                end
                flag = 2; % reach the step tolerance, maybe not convergence
                break;
            end

            if strcmp(options.Display, 'iter')
                fprintf('current step %d, alpha = %g', i, alpha);
                fprintf(', nfval = %d,  maxd = %g, f=%8.6g, gnorm=%8.6g\n',  obj.NF, max(abs(x)), f, gnorm)
            end

            if gnorm < options.NormGradTolerance 
                fprintf('The norm of current gradient is %g, which is smaller than the tolerance %g\n', gnorm, options.NormGradTolerance);
                flag = 1; % convergence 
                break;
            end

            s = alpha*d;
            y = g - pg; 
            sty = s'*y; 
            if sty < 0
                fprintf('bfgs: sty <= 0, skipping BFGS update at iteration %d \n', i)
            else
                if i < options.NumGrad
                    obj.S = [obj.S s];
                    obj.Y = [obj.Y y];
                else
                    obj.S = [obj.S(:, 2:end) s];
                    obj.Y = [obj.Y(:, 2:end) y];
                end
            end
        end

        if flag == 0
            flag = 3 % reach the max iteration, maybe not convergence.
        end
        fprintf('\n\n');
    end

    function r = hessian_gradient_prod(obj, g)
    %% LBFGS 更新
        N = size(obj.S, 2);
        q = g;
        for i = N:-1:1
            s = obj.S(:, i);
            y = obj.Y(:, i);
            rho(i) = 1/(s'*y);
            alpha(i) = rho(i)*(s'*q);
            q = q - alpha(i)*y;
        end

        r = obj.L'\(obj.L\(q));

        for i=1:N
            s = obj.S(:, i);
            y = obj.Y(:, i);
            beta = rho(i)*(y'*r);
            r = r + (alpha(i) - beta)*s;
        end
    end

    function [alpha, x, f, g] = line_search(obj, x0, f, g, d, fun, alpha)
    %% 用　Newton 方法求极小值点
        b = norm(g);
        a0 = 0;

        if nargin > 6
            a1 = alpha;
        else
            a1 = 2;
        end

        f0 = f;
        g0 = g'*d;

        x = x0 + a1*d;
        [f, g] = fun(x);
        obj.NF = obj.NF + 1;
        f1 = f;
        g1 = g'*d;

        k = 0;
        while k < 100 
            k = k + 1;
            if abs(a1 - a0) > 1e-15 & a1 > 0 
                t = g1 - (f1 - f0)/(a1 - a0);
                a2 = a1 - 0.5*(a1 - a0)*g1/t;
                x = x0 + a2*d;
                [f, g] = fun(x);
                obj.NF = obj.NF + 1;
                a0 = a1;
                f0 = f1;
                g0 = g1;

                f1 = f;
                g1 = g'*d;
                a1 = a2;
            else
                alpha = a1;
                break;
            end
        end
    end


    function  showlinesearch(obj, x0, f0, g0, d, fun, a, b)
    %% 显示线搜索的函数图像
        t = a:(b-a)/200:b;
        nf = length(t);
        F = zeros(nf, 1);
        for i = 1:nf
            [F(i), g] = fun(x0 + t(i)*d);
        end
        plot(t, F);
    end

    function [a, b] = search_interval(obj, x0, f0, g0, d, fun, alpha)
    %% 用进退法确定搜索区间
        a0 = 0;
        h0 = alpha;
        t = 2;
        k = 0;

        while true
            a1 = a0 + h0;
            [f1, g1] = fun(x0 + a1*d);
            if f1 < f0
                h0 = t*h0;
                alpha = a1;
                a0 = a1;
                f0 = f1;
            else
                a = min(alpha, a1);
                b = max(alpha, a1);
                break;
            end
        end
    end
end
end
