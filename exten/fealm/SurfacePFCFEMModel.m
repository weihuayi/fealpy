classdef SurfacePFCFEMModel < handle
    properties
        model
        space
        surface
        phi0
        phi1
        phi2
        timeline
        A
        M
    end
    methods
        function obj = SurfacePFCFEMModel(model, mesh, surface, timeline)
            obj.model = model;
            obj.space = LagrangeFiniteElementSpace(mesh);
            obj.surface = surface;
            obj.timeline = timeline;

            gdof = obj.space.number_of_global_dofs();
            NL = obj.timeline.get_number_of_time_levels();
            obj.phi0 = zeros(gdof, NL);
            obj.phi1 = zeros(gdof, NL);
            obj.phi2 = zeros(gdof, NL);
            obj.A = obj.space.stiff_matrix();
            obj.M = obj.space.mass_matrix();
        end

    end
end
