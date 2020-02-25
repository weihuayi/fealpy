classdef SpherePFCModelData < handle
properties
    r
    surface
end
methods
    function obj = SpherePFCModelData(r, C, R)
        obj.r = r;
        obj.surface = SphereSurface(C, R);
    end

    function mesh = init_mesh(obj)
        mesh = obj.surface.init_mesh();
    end

    function val = nonlinear_term(obj, phi)
        val = 0.5*(1 + obj.r)*phi.^2 + 0.25*phi.^4;
    end

    function val = grad_nonlinear_term(obj, phi)
        val = (1 + obj.r)*phi + phi.^3;
    end
end
end
