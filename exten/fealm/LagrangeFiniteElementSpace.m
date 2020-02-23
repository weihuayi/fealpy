classdef LagrangeFiniteElementSpace < handle
properties
    p
    q
    mesh
    cellmeasure
    spacetype

    TD % toplogy dimension
    GD % geometry dimentsion 
    name
end
methods
    function obj = LagrangeFiniteElementSpace(mesh, p, spacetype, q)
        if nargin == 1
            p = 1;
            spacetype = 'C';
            q = p + 2;
        elseif nargin == 2
            spacetype = 'C';
            q = p + 2;
        elseif nargin == 3
            q = p + 2;
        end

        obj.mesh = mesh;
        obj.p = p;
        obj.spacetype = spacetype;
        obj.q = q;

        obj.TD = mesh.top_dimension();
        obj.GD = mesh.geo_dimension();

        obj.cellmeasure = mesh.entity_measure('cell');
    end

    function TD = top_dimension(obj)
        TD = obj.TD;
    end

    function GD = geo_dimension(obj)
        GD = obj.GD;
    end

    function ldof = number_of_local_dofs(obj)
        p = obj.p;
        ldof = (p+1)*(p+2)/2; 
    end

    function gdof = number_of_global_dofs(obj)
        gdof = obj.mesh.number_of_nodes();
    end

    function cell2dof = cell_to_dof(obj)
        cell2dof = obj.mesh.entity('cell');
    end

    function ip = interpolation_points(obj)
        ip = obj.mesh.entity('node');
    end

    function uI = interpolation(obj, u)
        ip = obj.interpolation_points();
        uI = obj.get_function();
        uI.dofs = u(ip);
    end

    function uh = get_function(obj, dim)
        if nargin == 1
            dim = 1;
        end
        uh = Function(dim, obj);
    end

    function phi = basis(obj, bc)
        phi = bc;
    end

    function gphi = grad_basis(obj, bc)
        gphi = obj.mesh.grad_lambda()
    end

    function A = stiff_matrix(obj)
        ldof = obj.number_of_local_dofs()
        gdof = obj.number_of_global_dofs();
        cell2dof = obj.cell_to_dof();
        gphi = obj.grad_basis([1/3, 1/3, 1/3]);
        A = sparse(gdof, gdof);
        for i = 1:ldof
            for j = i:ldof
                Aij = dot(gphi(:, i), gphi(:, j), 2).*obj.cellmeasure;
                if (j==i)
                    A = A + sparse(cell2dof(:,i), cell2dof(:,j), Aij, gdof, gdof);
                else
                    A = A + sparse([cell2dof(:, i);cell2dof(:, j)], [cell2dof(:, j);cell2dof(:, i)],...
                                   [Aij; Aij], gdof, gdof);
                end
            end
        end
    end

    function M = mass_matrix(obj)
        ldof = obj.number_of_local_dofs()
        gdof = obj.number_of_global_dofs();
        cell2dof = obj.cell_to_dof();
        M = sparse(gdof, gdof);
        for i = 1:ldof
            for j = 1:ldof
                   Mij = obj.cellmeasure*((i==j)+1)/12;
                   M = M + sparse(cell2dof(:, i), cell2dof(:, j), Mij, gdof, gdof);             
            end
        end
    end

    function F = cross_matrix(obj, uh, cfun)
        ldof = obj.number_of_local_dofs();
        gdof = obj.number_of_global_dofs();
        cell2dof = obj.cell_to_dof();

        qf = TriangleQuadrature(4);
        NQ = qf.number_of_quad_points();
        NC = size(cell2dof, 1);
        Mij = zeros(NC, ldof, ldof); % the matrix on local cell
        for q = 1:NQ 
            [bc, w] = qf.get_quad_point_and_weight(q);
            phi = obj.basis(bc);
            val = cfun(obj.value(uh, bc));
            for i = 1:ldof
                for j = 1:ldof
                    Mij(:, i, j) = Mij(:, i, j) + w*phi(i)*phi(j)*val;
                end
            end
        end

        % construct the global matrix
        F = sparse(gdof, gdof);
        for i = 1:ldof
            for j = 1:ldof
                F = F + sparse(cell2dof(:, i), cell2dof(:, j), Mij(:, i, j).*obj.cellmeasure, gdof, gdof);             
            end
        end
    end


    function val = value(obj, uh, bc)
        dim = size(uh, 2);
        NC = obj.mesh.number_of_cells();
        ldof = obj.number_of_local_dofs();
        gdof = obj.number_of_global_dofs();
        phi = obj.basis(bc);
        cell2dof = obj.cell_to_dof();
        val = zeros(NC, dim);
        for i=1:ldof
            val = val + phi(i)*uh(cell2dof(:, i), :);
        end
    end

    function val = grad_value(obj, uh, bc)
        GD = obj.geo_dimension();
        dim = size(uh, 2);
        NC = obj.mesh.number_of_cells();
        ldof = obj.number_of_local_dofs();
        gdof = obj.number_of_global_dofs();
        gphi = obj.grad_basis(bc);
        cell2dof = obj.cell_to_dof();
        val = zeros(NC, GD, dim);
        for d=1:dim
            for i=1:ldof
                val(:, :, d) = val(:, :, d) + gphi(:, :, i).*uh(cell2dof(:, i), d);
            end
        end
    end
end
end
