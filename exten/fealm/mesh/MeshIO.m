classdef MeshIO
    properties
    end
    methods
        function c = read_inp(obj, f)
            fid = fopen(f, 'r');
            c = containers.Map;
            
            disp('Read heading:');
            line = fgetl(fid);
            while isempty(strfind(line, 'Preprint'))
                line = fgetl(fid);
            end
            
            disp('Reading part instances:');
            while true
                line = fgetl(fid);
                k = strfind(line, 'PART INSTANCE:');
                while isempty(k)
                    line = fgetl(fid);
                    k = strfind(line, 'PART INSTANCE:');
                end
                name = line(19:end);
                
                disp(['Reading part instance:', name]);
                
                line = fgetl(fid);
                line = fgetl(fid);
                disp(line);
                
                % Read Node Data
                data = textscan(fid, '%d %f %f %f', 'Delimiter', ',');
                node = cell2mat({data{2:end}});
                nodeid = data{1};
                start = nodeid(1) - 1;
                
                % Read Cell Data
                line = fgetl(fid);
                disp(line);
                data = textscan(fid, '%d %d %d %d %d', 'Delimiter', ',');
                cellid = data{1};
                cell = cell2mat({data{2:end}})' - start;
                c(name) = TetrahedronMesh(node, cell);  
                
                line = fgetl(fid);
                while ~strcmp(line(1), ',')
                    line = fgetl(fid);
                end
                line = fgetl(fid);
                disp(line);
                k = strfind(line, 'System');
                if ~isempty(k)
                    break;
                end
            end
            fclose(fid);
        end
        
        function fid = read_inp_head(obj, fid)
            line = fgetl(fid);
            while ~isempty(strfind(line, 'Preprint'))
            end
        end
        
        function [name, fid] = read_inp_part_name(obj, fid)
            line = fgetl(fid);
            while isempty(strfind(line, '-----'))
                line = fgetl(fid);
            end
            fgetl(fid);
            line = fgetl(fid);
            name = line(19:end);
        end
        
        function part = read_inp_part(obj, fid)
            % Read Node Data
            data = textscan(fid, '%d %f %f %f', 'Delimiter', ',');
            node = cell2mat({data{2:end}});
            nodeid = data{1};
            
            % Read Cell Data
            line = fgetl(fid);
            data = textscan(fid, '%d %d %d %d %d', 'Delimiter', ',');
            cellid = data{1};
            cell = cell2mat({data{2:end}})';
            part = TetrahedronMesh(node, cell);
            part.nodeid = nodeid;
            part.cellid = cellid;
        end
    end
end