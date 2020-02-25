function h = showmesh(node,elem,varargin)
%% SHOWMESH displays a triangular mesh in 2-D.
%
%    showmesh(node,elem) displays a topological 2-dimensional mesh,
%    including planar meshes and surface meshes. The mesh is given by node
%    and elem matrices; see <a href="matlab:ifem('meshdoc')">meshdoc</a> for the mesh data structure: 
%    node and elem.
%
%    showmesh(node,elem,viewangle) changes the display angle. The
%    deault view angle on planar meshes is view(2) and view(3) for surface
%    meshes. 
%        
%    showmesh(node,elem,'param','value','param','value'...) allows
%    additional patch param/value pairs to be used when displaying the
%    mesh.  For example, the default transparency parameter for a surface
%    mesh is set to 0.75. You can overwrite this value by using the param
%    pair ('FaceAlpha', value). The value has to be a number between 0 and
%    1. Other parameters include: 'Facecolor', 'Edgecolor' etc. These
%    parameters are mostly used when displaying a surface mesh.
%
%    To display a 3-dimensional mesh, use showmesh3 or showboundary3.
% 
%   Example:
%     % A mesh for a L-shaped domain
%     [node,elem] = squaremesh([-1,1,-1,1],0.5);
%     [node,elem] = delmesh(node,elem,'x>0 & y<0');
%     figure;
%     showmesh(node,elem);
%
%     % A mesh for a unit sphere
%     node = [1,0,0; 0,1,0; -1,0,0; 0,-1,0; 0,0,1; 0,0,-1];
%     elem = [6,1,2; 6,2,3; 6,3,4; 6,4,1; 5,1,4; 5,3,4; 5,3,2; 5,2,1];
%     for i = 1:3
%         [node,elem] = uniformrefine(node,elem);
%     end
%     r = sqrt(node(:,1).^2 + node(:,2).^2 + node(:,3).^2);
%     node = node./[r r r];
%     figure;
%     subplot(1,2,1);
%     showmesh(node,elem);
%     subplot(1,2,2);
%     showmesh(node,elem,'Facecolor','y','Facealpha',0.5);
%
%   See also showmesh3, showsolution, showboundary3.
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

dim = size(node,2);
nv = size(elem,2);
if (dim==2) && (nv==3) % planar triangulation
    h = trisurf(elem(:,1:3),node(:,1),node(:,2),zeros(size(node,1),1));
    set(h,'facecolor',[0.5 0.9 0.45],'edgecolor','k');
    view(2); axis equal; axis tight; axis off;
end
if (dim==2) && (nv==4) % planar quadrilateration
    h = patch('Faces', elem, 'Vertices', node);
    set(h,'facecolor',[0.5 0.9 0.45],'edgecolor','k');
    view(2); axis equal; axis tight; axis off;
end
if (dim==3) 
    if size(elem,2) == 3 % surface meshes
        h = trisurf(elem(:,1:3),node(:,1),node(:,2),node(:,3));    
        set(h,'facecolor',[0.5 0.9 0.45],'edgecolor','k','FaceAlpha',0.75);    
        view(3); axis equal; axis off; axis tight;    
    elseif size(elem,3) == 4
        showmesh3(node,elem,varargin{:});
        return
    end
end 
if (nargin>2) && ~isempty(varargin) % set display property
    if isnumeric(varargin{1})
        view(varargin{1});
        if nargin>3
            set(h,varargin{2:end});
        end
    else
        set(h,varargin{1:end});        
    end
end