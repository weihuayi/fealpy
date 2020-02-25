classdef Quadrature < handle
    methods
        function NQ = number_of_quad_points(obj)
            NQ = size(obj.quadpts, 1);
        end

        function [bc, w] = get_quad_point_and_weight(obj, i)
            bc = obj.quadpts(i, :);
            w = obj.weights(i);
        end
    end
end
