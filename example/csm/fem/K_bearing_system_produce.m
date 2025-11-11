function [K_bearing_system]=K_bearing_system_produce(order_load,component_order_node,order_section_shaft,order_bearing_vector,bearing_layout,bearing_stiffness_matrix_spectrum)
%%%参数输出
%轴承（除概念轴承外）全局坐标系的联结附加的刚度矩阵，K_bearing_system，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=K_bearing_system*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
%%%参数输入
%工况序号，order_load
%异形件的节点序号信息矩阵，component_order_node，[1 异形件序号 2 作用类型（1表示接地，2表示替代轴） 3 激活状态（0表示未激活，1表示激活） 4 装配完整（0表示完整，1表示不完整） 5 节点数量（n） 6至n+5 节点序号]
%轴系的轴序号及截面信息矩阵，order_section_shaft，第1行为轴序号，第2行为对应轴的截面数
%轴承（除概念轴承外）序号向量，order_bearing_vector
%轴系的轴承装配参数矩阵，bearing_layout
%球轴承/滚子轴承 [1 轴承序号 2 与其内圈联结轴的序号 3 轴承内圈节点序号 4 轴承中心距内圈联结轴左端距离（mm） 5 与其外圈联结轴的序号 6 轴承外圈节点序号 7 轴承中心距外圈联结轴左端距离（mm） 8 轴承额定动载荷(N)，capacity_dynamic 9 轴承额定静载荷(N)，capacity_static 10-12 空 13 滚动体列数，bearing_element_column 14 轴承类型，bearing_type 15 轴承质量(kg)，bearing_mass 16 轴承宽度(mm)，bearing_width 17 轴承内圈(mm)，bearing_inner 18 轴承外圈(mm)，bearing_outer 19 球轴承初始接触角/滚子轴承接触角(rad)，contact_angle_initial 20 球轴承内圈沟曲率，bearing_fi 21 球轴承外圈沟曲率，bearing_fe 22 滚动体个数，bearing_element_number 23 滚动体直径/锥轴承滚动体中径(mm)，bearing_element_diameter 24 锥轴承滚动体锥角(rad)，taper_angle 25 滚子轴承滚动体长度(mm)，bearing_element_length 26 滚子轴承圆角半径(mm)，bearing_element_fillet 27 轴承径向游隙(mm)，bearing_radial_clearance 28 轴承轴向游隙(mm)，bearing_axial_clearance 29 内圈挡边位置（0表示左边，1表示右边），bearing_rib_position 30 圆柱滚子内圈左边是否带档边，bearing_rib_inner_left 31 圆柱滚子内圈右边是否带档边，bearing_rib_inner_right 32 圆柱滚子外圈左边是否带档边，bearing_rib_outer_left 33 圆柱滚子外圈右边是否带档边，bearing_rib_outer_right 34 滚动体修形类型，type_element_modi 35 轴承外圈弹性模量(Mpa)，E_outer 36 轴承外圈泊松比，v_outer 37 轴承滚动体弹性模量(Mpa),E_element 38 轴承滚动体泊松比，v_element 39 轴承内圈弹性模量(Mpa)，E_inner 40 轴承内圈泊松比，v_inner 41 滚动体切片个数]
%概念轴承 [1 轴承序号 2 与其内圈联结轴的序号 3 轴承内圈节点序号 4 轴承中心距内圈联结轴左端距离（mm） 5 与其外圈联结轴的序号 6 轴承外圈节点序号 7 轴承中心距外圈联结轴左端距离（mm） 8 轴承额定动载荷(N)，capacity_dynamic 9 轴承额定静载荷(N)，capacity_static 10-13 空 14 轴承类型，bearing_type 15 轴承质量(kg)，bearing_mass 16 轴承宽度(mm)，bearing_width 17 轴承内圈(mm)，bearing_inner 18 轴承外圈(mm)，bearing_outer 19-41 空]
%系统全工况所有轴承的全局坐标系的刚度矩阵，bearing_stiffness_matrix_spectrum，[工况序号 轴承序号（5*1） 轴承刚度矩阵（5*5）]，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm)]'=轴承刚度矩阵*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad)]'
%%%程序开始
%计算轴系所有轴的最大截面数，sum_shaft_segment
sum_shaft_segment=sum(order_section_shaft(2,:));
%%%计算轴承（除概念轴承外）联结附加的刚度矩阵，K_bearing_system，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=K_bearing_system*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
%定义轴承（除概念轴承外）联结附加的刚度矩阵，K_bearing_system，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=K_bearing_system*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
K_bearing_system=zeros(6*sum_shaft_segment,6*sum_shaft_segment);
if isempty(bearing_stiffness_matrix_spectrum)~=1
    %记录第order_load工况的所有轴承的全局坐标系的刚度矩阵，bearing_stiffness_matrix_load，[工况序号 轴承序号（5*1） 轴承刚度矩阵（5*5）]，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm)]'=轴承刚度矩阵*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad)]'
    id_bearing_load=(bearing_stiffness_matrix_spectrum(:,1)==order_load);  %工况序号，order_load
    bearing_stiffness_matrix_load=bearing_stiffness_matrix_spectrum(id_bearing_load,:);
    %%%轴承（除概念轴承外）个数，number_bearing
    size_bearing_vector=size(order_bearing_vector);  %轴承（除概念轴承外）序号向量，order_bearing_vector
    number_bearing=size_bearing_vector(1,1);
    %计算轴承（除概念轴承外）联结附加的刚度矩阵，K_bearing_system，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=K_bearing_system*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    if number_bearing>0  %轴承（除概念轴承外）个数，number_bearing
        for ii=1:number_bearing  %轴承（除概念轴承外）个数，number_bearing
            %记录第ii个轴承（除概念轴承外）联结的参数
            %记录第ii个轴承（除概念轴承外）的序号，order_bearing
            order_bearing=order_bearing_vector(ii,1);
            %记录第ii个轴承（除概念轴承外）装配参数矩阵，bearing_layout_temp
            %球轴承/滚子轴承 [1 轴承序号 2 与其内圈联结轴的序号 3 轴承内圈节点序号 4 轴承中心距内圈联结轴左端距离（mm） 5 与其外圈联结轴的序号 6 轴承外圈节点序号 7 轴承中心距外圈联结轴左端距离（mm） 8 轴承额定动载荷(N)，capacity_dynamic 9 轴承额定静载荷(N)，capacity_static 10-12 空 13 滚动体列数，bearing_element_column 14 轴承类型，bearing_type 15 轴承质量(kg)，bearing_mass 16 轴承宽度(mm)，bearing_width 17 轴承内圈(mm)，bearing_inner 18 轴承外圈(mm)，bearing_outer 19 球轴承初始接触角/滚子轴承接触角(rad)，contact_angle_initial 20 球轴承内圈沟曲率，bearing_fi 21 球轴承外圈沟曲率，bearing_fe 22 滚动体个数，bearing_element_number 23 滚动体直径/锥轴承滚动体中径(mm)，bearing_element_diameter 24 锥轴承滚动体锥角(rad)，taper_angle 25 滚子轴承滚动体长度(mm)，bearing_element_length 26 滚子轴承圆角半径(mm)，bearing_element_fillet 27 轴承径向游隙(mm)，bearing_radial_clearance 28 轴承轴向游隙(mm)，bearing_axial_clearance 29 内圈挡边位置（0表示左边，1表示右边），bearing_rib_position 30 圆柱滚子内圈左边是否带档边，bearing_rib_inner_left 31 圆柱滚子内圈右边是否带档边，bearing_rib_inner_right 32 圆柱滚子外圈左边是否带档边，bearing_rib_outer_left 33 圆柱滚子外圈右边是否带档边，bearing_rib_outer_right 34 滚动体修形类型，type_element_modi 35 轴承外圈弹性模量(Mpa)，E_outer 36 轴承外圈泊松比，v_outer 37 轴承滚动体弹性模量(Mpa),E_element 38 轴承滚动体泊松比，v_element 39 轴承内圈弹性模量(Mpa)，E_inner 40 轴承内圈泊松比，v_inner 41 滚动体切片个数]
            id_bearing=(bearing_layout(:,1)==order_bearing);  %第ii个轴承（除概念轴承外）的序号，order_bearing
            %记录第ii个轴承（除概念轴承外）的内圈联结轴的序号
            order_inner=bearing_layout(id_bearing,2);
            %记录第ii个轴承（除概念轴承外）的内圈节点的序号
            node_inner=bearing_layout(id_bearing,3);
            %记录第ii个轴承（除概念轴承外）的外圈联结轴的序号
            order_outer=bearing_layout(id_bearing,5);
            %记录第ii个轴承（除概念轴承外）的外圈节点的序号
            node_outer=bearing_layout(id_bearing,6);
            %%%计算第ii个轴承内圈节点在所有轴截面的位置，Nbs_inner
            if order_inner>0  %轴承内圈联接轴
                position_inner=find(roundn(order_section_shaft(1,:),-8)==roundn(order_inner,-8));  %轴系的轴序号及截面信息矩阵，order_section_shaft，第1行为轴序号，第2行为对应轴的截面数
                if position_inner==1
                    Nbs_inner=node_inner-1;  %第ii个轴承内圈节点在所有轴截面的位置
                else
                    Nbs_inner=sum(order_section_shaft(2,1:position_inner-1))+node_inner-1;  %第ii个轴承内圈节点在所有轴截面的位置
                end
            elseif order_inner<0  %轴承内圈联接异形件
                %%%计算轴承内圈节点在异形件的位置，position_node_inner
                %记录序号为order_inner异形件的节点序号信息，component_order_node_inner
                id_component_inner=(roundn(component_order_node(:,1),-8)==roundn(order_inner,-8));  %异形件的节点序号信息矩阵，component_order_node，[1 异形件序号 2 作用类型（1表示接地，2表示替代轴） 3 激活状态（0表示未激活，1表示激活） 4 装配完整（0表示完整，1表示不完整） 5 节点数量（n） 6至n+5 节点序号]
                %记录异形件的节点数量，number_node_inner
                number_node_inner=component_order_node(id_component_inner,5);
                %记录异形件的节点序号行向量，component_order_node_inner
                component_order_node_inner=component_order_node(id_component_inner,6:number_node_inner+5);
                %计算轴承内圈节点在异形件的位置，position_node_inner
                position_node_inner=find(component_order_node_inner(1,:)==node_inner);  %第ii个轴承（除概念轴承外）的内圈节点的序号
                %%%计算轴承内圈节点在所有轴系节点的位置，position_inner
                position_inner=find(roundn(order_section_shaft(1,:),-8)==roundn(order_inner,-8));  %轴系的轴/异形件序号及节点信息矩阵，order_section_shaft，第1行为轴/异形件序号，第2行为对应轴/异形件的节点总数
                if position_inner==1
                    Nbs_inner=position_node_inner-1;
                else
                    Nbs_inner=sum(order_section_shaft(2,1:position_inner-1))+position_node_inner-1;  %轴系的轴/异形件序号及节点信息矩阵，order_section_shaft，第1行为轴/异形件序号，第2行为对应轴/异形件的节点总数
                end
            end
            %%%
            %%%计算第ii个轴承外圈节点在所有轴截面的位置，Nbs_outer
            if order_outer>0  %轴承外圈联接轴
                position_outer=find(roundn(order_section_shaft(1,:),-8)==roundn(order_outer,-8));  %轴系的轴序号及截面信息矩阵，order_section_shaft，第1行为轴序号，第2行为对应轴的截面数
                if position_outer==1
                    Nbs_outer=node_outer-1;  %第ii个轴承外圈节点在所有轴截面的位置
                else
                    Nbs_outer=sum(order_section_shaft(2,1:position_outer-1))+node_outer-1;  %第ii个轴承外圈节点在所有轴截面的位置
                end
            elseif order_outer<0  %轴承外圈联接异形件
                %%%计算轴承外圈节点在异形件的位置，position_node_outer
                %记录序号为order_outer异形件的节点序号信息，component_order_node_outer
                id_component_outer=(roundn(component_order_node(:,1),-8)==roundn(order_outer,-8));  %异形件的节点序号信息矩阵，component_order_node，[1 异形件序号 2 作用类型（1表示接地，2表示替代轴） 3 激活状态（0表示未激活，1表示激活） 4 装配完整（0表示完整，1表示不完整） 5 节点数量（n） 6至n+5 节点序号]
                %记录异形件的节点数量，number_node_outer
                number_node_outer=component_order_node(id_component_outer,5);
                %记录异形件的节点序号行向量，component_order_node_outer
                component_order_node_outer=component_order_node(id_component_outer,6:number_node_outer+5);
                %计算轴承外圈节点在异形件的位置，position_node_outer
                position_node_outer=find(component_order_node_outer(1,:)==node_outer);  %第ii个轴承（除概念轴承外）的外圈节点的序号
                %%%计算轴承外圈节点在所有轴系节点的位置，position_outer
                position_outer=find(roundn(order_section_shaft(1,:),-8)==roundn(order_outer,-8));  %轴系的轴/异形件序号及节点信息矩阵，order_section_shaft，第1行为轴/异形件序号，第2行为对应轴/异形件的节点总数
                if position_outer==1
                    Nbs_outer=position_node_outer-1;
                else
                    Nbs_outer=sum(order_section_shaft(2,1:position_outer-1))+position_node_outer-1;  %轴系的轴/异形件序号及节点信息矩阵，order_section_shaft，第1行为轴/异形件序号，第2行为对应轴/异形件的节点总数
                end
            end
            %记录第ii个轴承（除概念轴承外）的刚度矩阵，bearing_stiffness_matrix，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm)]'=bearing_stiffness_matrix*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad)]'
            %第order_load工况的所有轴承的全局坐标系的刚度矩阵，bearing_stiffness_matrix_load，[工况序号 轴承序号（5*1） 轴承刚度矩阵（5*5）]，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm)]'=轴承刚度矩阵*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad)]'
            id_bearing_stiffness=(roundn(bearing_stiffness_matrix_load(:,2),-8)==roundn(order_bearing,-8));  %第ii个轴承（除概念轴承外）的序号，order_bearing
            bearing_stiffness_matrix=bearing_stiffness_matrix_load(id_bearing_stiffness,3:7);
            %%%计算第ii个轴承（除概念轴承外）联结附加的刚度矩阵，K_bearing_system_temp，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=K_bearing_system_temp*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
            if (order_inner~=0)&&(order_outer==0)  %轴承内圈不接地，外圈接地
                K_bearing_system(6*Nbs_inner+1:6*Nbs_inner+5,6*Nbs_inner+1:6*Nbs_inner+5)=K_bearing_system(6*Nbs_inner+1:6*Nbs_inner+5,6*Nbs_inner+1:6*Nbs_inner+5)+bearing_stiffness_matrix;
            elseif (order_inner==0)&&(order_outer~=0)  %轴承内圈接地，外圈不接地
                K_bearing_system(6*Nbs_outer+1:6*Nbs_outer+5,6*Nbs_outer+1:6*Nbs_outer+5)=K_bearing_system(6*Nbs_outer+1:6*Nbs_outer+5,6*Nbs_outer+1:6*Nbs_outer+5)+bearing_stiffness_matrix;
            else  %轴承内圈不接地，外圈不接地
                K_bearing_system(6*Nbs_inner+1:6*Nbs_inner+5,6*Nbs_inner+1:6*Nbs_inner+5)=K_bearing_system(6*Nbs_inner+1:6*Nbs_inner+5,6*Nbs_inner+1:6*Nbs_inner+5)+bearing_stiffness_matrix;
                K_bearing_system(6*Nbs_outer+1:6*Nbs_outer+5,6*Nbs_outer+1:6*Nbs_outer+5)=K_bearing_system(6*Nbs_outer+1:6*Nbs_outer+5,6*Nbs_outer+1:6*Nbs_outer+5)+bearing_stiffness_matrix;
                K_bearing_system(6*Nbs_inner+1:6*Nbs_inner+5,6*Nbs_outer+1:6*Nbs_outer+5)=K_bearing_system(6*Nbs_inner+1:6*Nbs_inner+5,6*Nbs_outer+1:6*Nbs_outer+5)-bearing_stiffness_matrix;
                K_bearing_system(6*Nbs_outer+1:6*Nbs_outer+5,6*Nbs_inner+1:6*Nbs_inner+5)=K_bearing_system(6*Nbs_outer+1:6*Nbs_outer+5,6*Nbs_inner+1:6*Nbs_inner+5)-bearing_stiffness_matrix;
            end
        end
    end
    %%%强制对称矩阵
    K_bearing_system=0.5*(K_bearing_system+K_bearing_system');
end
%%%程序结束