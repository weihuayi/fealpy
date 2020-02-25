classdef Function < handle
properties
    dofs
    space
end
methods
    function obj = Function(dim, space)
        gdof = space.number_of_global_dofs();
        obj.dofs = zeros(gdof, dim);
        obj.space = space;
    end

    function val = value(obj, bc)
        val = obj.space.value(obj.dofs, bc);
    end
    function val = grad_value(obj, bc)
        val = obj.space.grad_value(obj.dofs, c);
    end
end
end
