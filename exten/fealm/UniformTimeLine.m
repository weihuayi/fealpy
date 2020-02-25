classdef UniformTimeLine < handle
    properties
       T0 % the start time
       T1 % the stop time
       NL % the number of time levels
       dt % the length of time step
       current % the index of current time levels
    end
    methods
        function obj=UniformTimeLine(T0, T1, NT)
            obj.T0 = T0;
            obj.T1 = T1;
            obj.NL = NT + 1;
            obj.dt = (T1 - T0)/NT;
            obj.current = 1;
        end

        function NL = get_number_of_time_levels(obj)
            NL = obj.NL;
        end

        function idx = get_current_time_level_index(obj)
            idx = obj.current;
        end

        function t = get_next_time_level(obj)
            t = obj.T0 + obj.dt*obj.current;
        end

        function dt = get_time_step_length(obj)
            dt = obj.dt;
        end

        function flag = stop(obj)
            flag = obj.current >= obj.NL;
        end

        function reset(obj)
            obj.current = 1;
        end
    end
end
