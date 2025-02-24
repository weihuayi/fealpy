%%%计算行星滚柱丝杠的载荷及接触应力分布
function [potential_total,Fn_nut,Fn_screw,flexible_inner,flexible_outer,B_inner,B_outer,half_width_a_inner,half_width_b_inner,stress_inner,half_width_a_outer,half_width_b_outer,stress_outer,ratio_nut,ratio_screw,distribution_nut,distribution_screw,delta_outer,delta_inner,S_load_nut,S_load_screw]=calulate_load_distribution_roller_screw(order_roller_screw,order_load,roller_screw_initial,type_load,norm_roller_screw,mesh_outer_matrix,mesh_inner_matrix,modify_nut,modify_screw)
%%%参数输出
%行星滚柱丝杠总的势能，potential_total
%螺母的接触点法向载荷的向量，Fn_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
%丝杠的接触点法向载荷的向量，Fn_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
%内螺纹的啮合柔度(mm/N)，flexible_inner
%外螺纹的啮合柔度(mm/N)，flexible_outer
%内啮合压缩势能辅助值，B_inner
%外啮合压缩势能辅助值，B_outer
%内啮合的接触长半轴(mm)，half_width_a_inner
%内啮合的接触短半轴(mm)，half_width_b_inner
%内啮合接触应力(Mpa)，stress_inner
%外啮合的接触长半轴(mm)，half_width_a_outer
%外啮合的接触短半轴(mm)，half_width_b_outer
%外啮合接触应力(Mpa)，stress_outer
%螺母轴向分量比例，ratio_nut
%丝杠轴向分量比例，ratio_screw
%螺母的接触点法向载荷不均载系数的向量，distribution_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
%螺母的接触点法向载荷不均载系数的向量，distribution_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
%内啮合接触变形量(mm)，delta_inner
%外啮合接触变形量(mm)，delta_outer
%螺母载荷分布均方值，S_load_nut
%丝杠载荷分布均方值，S_load_screw
%%%参数输入
%行星滚柱丝杠序号，order_roller_screw
%工况序号，order_load
%行星滚柱丝杠基本参数，roller_screw_initial，[1 行星滚柱丝杠序号 2 行星滚柱丝杠的类型(1表示标准式，2表示反向式),type_roller_screw 3 滚柱个数,number_roller 4 丝杠螺纹头数,N_thread_s 5 螺母螺纹头数,N_thread_n 6 滚柱螺纹头数,N_thread_r
%7 丝杠螺纹旋向(-1表示左旋，1表示右旋),direct_screw 8 螺母螺纹旋向(-1表示左旋，1表示右旋),direct_nut 9 滚柱螺纹旋向(-1表示左旋，1表示右旋),direct_roller 10 丝杠螺纹升角(rad)，rise_angle_s 11 螺母螺纹升角(rad)，rise_angle_n 12 滚柱螺纹升角(rad)，rise_angle_r
%13 第1个滚柱与X轴夹角(rad)，angle_roller 14 内螺纹当量摩擦系数，f_inner 15 外螺纹当量摩擦系数，f_outer 16 螺距(mm)，pitch_screw 17 牙型角(rad)，angle_screw 18 滚柱牙型圆弧半径(mm)，radius_roller 19 滚柱齿廓圆心截面坐标系X轴值(mm)，X_center_r 20 滚柱齿廓圆心截面坐标系Y轴值(mm)，Y_center_r
%21 丝杠螺纹削顶系数，cutter_top_s 22 丝杠螺纹削根系数，cutter_bottom_s 23 螺母螺纹削顶系数，cutter_top_n 24 螺母螺纹削根系数，cutter_bottom_n 25 滚柱螺纹削顶系数，cutter_top_r 26 滚柱螺纹削根系数，cutter_bottom_r 27 丝杠螺纹半牙减薄量(mm)，reduce_s 28 螺母螺纹半牙减薄量(mm)，reduce_n 29 滚柱螺纹半牙减薄量(mm)，reduce_r 30 空
%31 丝杠的螺纹齿顶高(mm)，addendum_s 32 丝杠的螺纹齿根高(mm)，dedendum_s 33 螺母的螺纹齿顶高(mm)，addendum_n 34 螺母的螺纹齿根高(mm)，dedendum_n 35 滚柱的螺纹齿顶高(mm)，addendum_r 36 滚柱的螺纹齿根高(mm)，dedendum_r
%37 丝杠实际顶径(mm)，D_top_s 38 丝杠实际根径(mm)，D_bottom_s 39 螺母实际顶径(mm)，D_top_n
%40 螺母实际根径(mm)，D_bottom_n 41 滚柱实际顶径(mm)，D_top_r 42 滚柱实际根径(mm)，D_bottom_r
%43 丝杠中径(mm),D_pitch_s ,44 螺母中径(mm),D_pitch_n, 45 滚柱中径(mm)，D_pitch_r
%46 丝杠轴长度(mm)，length_shaft_s 47 螺母轴长度(mm)，length_shaft_n 48 滚柱轴长度(mm)，length_shaft_r 49 丝杠轴外径(mm)，diameter_outer_s 50 螺母轴外径(mm)，diameter_outer_n 51 滚柱轴外径(mm)，diameter_outer_r 52 丝杠轴内径(mm)，diameter_inner_s 53 螺母轴内径(mm)，diameter_inner_n 54 滚柱轴内径(mm)，diameter_inner_r
%55 丝杠螺纹长度(mm)，length_thread_s 56 螺母螺纹长度(mm)，length_thread_n 57 滚柱螺纹长度(mm)，length_thread_r 58 内啮合的有效螺纹长度(mm)，length_mesh_inner 59 外啮合的有效螺纹长度(mm)，length_mesh_outer 60 空
%61 左齿轮宽度(mm)，width_gear_left 62 右齿轮宽度(mm)，width_gear_left 63 左保持架宽度(mm)，width_carrier_left 64 左保持架宽度(mm)，width_carrier_right
%65 螺母轴左端相对于丝杠轴左端的位置(mm)，delta_n_s 66 滚柱轴左端相对于丝杠轴左端的位置(mm)，delta_r_s 67 丝杠螺纹左端相对于丝杠轴左端的位置(mm),delta_thread_s 68 螺母螺纹左端相对于螺母轴左端的位置(mm),delta_thread_n 69 滚柱螺纹左端相对于滚柱轴左端的位置(mm),delta_thread_r 70 轴向预紧载荷(N)，preload_axial
%71 丝杠轴的密度(t/mm^-9)，density_s 72 螺母轴的密度(t/mm^-9)，density_n 73 滚柱轴的密度(t/mm^-9)，density_r 74 丝杠弹性模量(Mpa)，E_s 75 螺母弹性模量(Mpa)，E_n 76 滚柱弹性模量(Mpa)，E_r 77 丝杠泊松比，possion_s 78 螺母泊松比，possion_n 79 滚柱泊松比，possion_r
%80 左齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_left 81 右齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_right 82 左行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_left 83 右行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_right
%84 左保持架外径(mm)，outer_carrier_left 85 左保持架内径(mm)，inner_carrier_left 86 右保持架外径(mm)，outer_carrier_right 87 右保持架内径(mm)，inner_carrier_right 88 行星滚柱丝是否指定效率(0表示不指定，1表示指定),sign_efficiency 89 正向驱动效率(无预紧载荷)，advance_efficiency 90 逆向驱动效率(无预紧载荷)，reverse_efficiency
%91 齿轮法向模数(mm)，m_n 92 内齿圈齿数，z_r 93 太阳轮齿数，z_s 94 行星轮齿数，z_p 95 压力角(rad)，press_angle 96 端面压力角(rad)，tran_press_angle 97 螺旋角(rad)，helix_angle 98 内齿圈法向变位系数，n_r 99 太阳轮法向变位系数，n_s 100 行星轮法向变位系数，n_p
%101 工作中心距(mm)，work_center 102 行星轮系内啮合角，mesh_angle(rad) 103 行星轮系外啮合角，mesh_angle(rad) 104 内齿圈齿顶圆，tran_ad_dia_r（mm） 105 太阳轮齿顶圆，tran_ad_dia_s（mm） 106 行星轮齿顶圆，tran_ad_dia_p（mm） 107 内齿圈齿根圆，tran_de_dia_r（mm） 108 太阳轮齿根圆，tran_de_dia_s（mm） 109 行星轮齿根圆，tran_de_dia_p（mm）]
%110 内齿圈的节圆（mm），tran_pitch_dia_r 111 太阳轮的节圆（mm），tran_pitch_dia_s 112 行星轮的节圆（mm），tran_pitch_dia_p 113-120 空
%121 螺母幅板宽度(mm)，width_web_n 122 螺母幅板中心距螺母轴左端距离(mm)，delta_web_n 123 丝杠左幅板宽度(mm)，width_left_web_s 124 丝杠左幅板中心距丝杠轴左端距离(mm)，delta_left_web_s 125 丝杠右幅板宽度(mm)，width_right_web_s 126 丝杠右幅板中心距丝杠轴左端距离(mm)，delta_right_web_s 127-130 空
%131 螺母轮缘轴序号,shaft_self_nut 132 丝杠轮缘轴序号，shaft_self_screw 133 滚柱轮缘轴序号，shaft_self_roller,134-140 空
%141 行星轮刀具齿顶高系数，142 行星轮刀具顶隙系数，143 行星轮齿顶削减量，144 内齿圈刀具齿顶高系数，145 内齿圈刀具顶隙系数，146 内齿圈齿顶削减量,147 太阳轮刀具齿顶高系数，148 太阳轮刀具顶隙系数，149 太阳轮齿顶削减量
%150-160 空 161 有效内啮合螺纹相对于螺母轴左端的位置(mm)，thread_inner_nut 162 有效内啮合螺纹相对于滚柱轴左端的位置(mm)，thread_inner_roller 163 有效外啮合螺纹相对于丝杠轴左端的位置(mm)，thread_outer_screw 164 有效外啮合螺纹相对于滚柱轴左端的位置(mm)，thread_outer_roller]
%行星滚柱丝杠受载方式，type_load，1表示同侧受载，2表示异侧受载
%行星滚柱丝杠外螺纹/内螺纹的法向载荷相关矩阵，norm_roller_screw，[1 工况序号 2 行星滚柱丝杠序号 3 丝杠功率(kW) 4 螺母功率(kW) 5  无预紧效率
%6 丝杠轴序号 7 丝杠工作面(-1表示下轮廓，1表示上轮廓) 8 丝杠工作压力角(rad) 9 丝杠受载个数 10 丝杠法向载荷(N) 11 丝杠的法向矢量X轴分量 12 丝杠的法向矢量Y轴分量 13 丝杠的法向矢量Z轴分量
%14 螺母轴序号 15 螺母工作面(-1表示下轮廓，1表示上轮廓) 16 螺母工作压力角(rad) 17 螺母受载个数 18 螺母法向载荷(N) 19 螺母的法向矢量X轴分量 20 螺母的法向矢量Y轴分量 21 螺母的法向矢量Z轴分量
%22 旋转机构因预紧力的法向载荷 23 驱动形式(1表示正向驱动，2表示逆向驱动) 24 直线机构线速度(mm/s),velocity_linear 25 旋转机构自转速度(rpm),speed_turn 26 滚柱相对转动速度(rpm),speed_roller 27 持续时间(s),time 28 循环次数，number_cycle 29 行星滚柱丝杠的类型(1表示标准式，2表示反向式),type_roller_screw]
%行星滚柱丝杠螺纹外啮点位置矩阵，mesh_outer_matrix，[1 行星滚柱丝杠序号 2 丝杠工作面(-1表示下轮廓，1表示上轮廓) 3 滚柱工作面(-1表示下轮廓，1表示上轮廓) 4 外啮合点全局坐标系X轴坐标(mm) 5 外啮合点全局坐标系Y轴坐标(mm) 6 丝杠外啮合点夹角(rad) 7 丝杠外啮合点半径(mm) 8 滚柱外啮合夹角(rad) 9 外啮合点半径(mm) 10 外啮点轴向间隙(mm)]
%行星滚柱丝杠螺纹内啮点位置矩阵，mesh_inner_matrix，[1 行星滚柱丝杠序号 2 螺母工作面(-1表示下轮廓，1表示上轮廓) 3 滚柱工作面(-1表示下轮廓，1表示上轮廓) 4 内啮合点全局坐标系X轴坐标(mm) 5 内啮合点全局坐标系Y轴坐标(mm) 6 螺母内啮合点夹角(rad) 7 螺母内啮合点半径(mm) 8 滚柱内啮合夹角(rad) 9 内啮合点半径(mm) 10 内啮点轴向间隙(mm)]
%螺母侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(um)，modify_nut
%丝杠侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(um)，modify_screw
%%%程序开始
%%%记录赫兹接触系数表，H
H=[0,0.02363,0.02363,0.0002791;0.05,0.02444,0.02286,0.000279;0.1,0.02529,0.02212,0.0002785;0.15,0.0262,0.02142,0.0002777;0.2,0.02717,0.02074,0.0002766;0.25,0.02821,0.02008,0.0002751;0.3,0.02934,0.01943,0.0002733;0.35,0.03057,0.0188,0.0002711;0.4,0.03192,0.01818,0.0002685;0.45,0.03342,0.01756,0.0002654;0.5,0.03511,0.01694,0.0002617;0.55,0.03703,0.01632,0.0002574;0.6,0.03924,0.01569,0.0002525;0.62,0.04023,0.01544,0.0002502;0.64,0.04130,0.01517,0.0002479;0.66,0.04244,0.01491,0.0002453;0.68,0.04367,0.01464,0.0002426;0.7,0.04502,0.01437,0.0002397;0.71,0.04573,0.01423,0.0002381;0.72,0.04649,0.01408,0.0002365;0.73,0.04726,0.01394,0.0002349;0.74,0.04809,0.01380,0.0002331;0.75,0.04896,0.01365,0.0002313;0.76,0.04988,0.01350,0.0002294;0.77,0.05086,0.01334,0.0002275;0.78,0.05189,0.01319,0.0002254;0.79,0.05299,0.01303,0.0002233;0.8,0.05416,0.01286,0.000221;0.805,0.05477,0.01278,0.0002198;0.81,0.05540,0.01269,0.0002186;0.815,0.05606,0.01261,0.0002174;0.82,0.05674,0.01252,0.0002162;0.825,0.05744,0.01244,0.0002149;0.83,0.05817,0.01235,0.0002136;0.835,0.05895,0.01225,0.0002122;0.840,0.05974,0.01216,0.0002108;0.845,0.06058,0.01207,0.0002093;0.85,0.06144,0.01197,0.0002079;0.855,0.06236,0.01187,0.0002063;0.86,0.06330,0.01178,0.0002047;0.865,0.06430,0.01167,0.0002031;0.87,0.06537,0.01157,0.0002014;0.875,0.06648,0.01146,0.0001996;0.88,0.06767,0.01135,0.0001979;0.885,0.06890,0.01124,0.0001959;0.89,0.07017,0.01112,0.000194;0.895,0.07161,0.01101,0.0001919;0.9,0.07305,0.01089,0.0001898;0.905,0.07469,0.01076,0.0001875;0.91,0.07641,0.01063,0.0001852;0.915,0.07823,0.01050,0.0001828;0.92,0.08023,0.01036,0.0001802;0.925,0.08247,0.01021,0.0001775;0.93,0.08480,0.01006,0.0001747;0.935,0.08743,0.009898,0.0001716;0.94,0.09035,0.009729,0.0001684;0.945,0.09362,0.009550,0.0001649;0.95,0.09733,0.009359,0.0001611;0.955,0.1016,0.009153,0.000157;0.96,0.1066,0.008930,0.0001525;0.965,0.1124,0.008690,0.0001476;0.970,0.1197,0.008414,0.0001419;0.975,0.1284,0.008118,0.0001357;0.98,0.1404,0.007757,0.0001281;0.985,0.1573,0.007323,0.0001189;0.99,0.1831,0.006784,0.0001074;0.995,0.2398,0.005923,0.00008911];
%%%计算行星滚柱丝杠的载荷及应力分布
if sum(roller_screw_initial(:,1)==order_roller_screw)>0 %行星滚柱丝杠序号，order_roller_screw
    %%%调整螺母侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_nut
    modify_nut=10^(-3)*modify_nut;
    modify_nut=modify_nut-min(modify_nut);
    %%%调整丝杠侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_screw
    modify_screw=10^(-3)*modify_screw;
    modify_screw=modify_screw-min(modify_screw);
    %%%记录序号为order_roller_screw的行星滚柱丝杠的基本参数，roller_screw_temp
    id_roller_screw=(roller_screw_initial(:,1)==order_roller_screw);
    roller_screw_temp=roller_screw_initial(id_roller_screw,:);
    %%%记录行星滚柱丝杠的(1表示标准式，2表示反向式),type_roller_screw
    type_roller_screw=roller_screw_temp(1,2);
    %%%记录行星滚柱丝杠的滚柱个数，number_roller
    number_roller=roller_screw_temp(1,3);
    %%%记录行星滚柱丝杠的螺矩(mm)，pitch_screw
    pitch_screw=roller_screw_temp(1,16);
    %%%记录滚柱牙型圆弧半径(mm)，radius_roller
    radius_roller=roller_screw_temp(1,18);
    %%%记录丝杠实际根径(mm)，D_bottom_s
    D_bottom_s=roller_screw_temp(1,38);
    %%%记录螺母实际根径(mm)，D_bottom_n
    D_bottom_n=roller_screw_temp(1,40);
    %%%记录滚柱实际根径(mm)，D_bottom_r
    D_bottom_r=roller_screw_temp(1,42);
    %%%记录螺母轴外径(mm)，diameter_outer_n
    diameter_outer_n=roller_screw_temp(1,50);
    %%%记录丝杠轴内径(mm)，diameter_inner_s
    diameter_inner_s=roller_screw_temp(1,52);
    %%%记录滚柱轴内径(mm)，diameter_inner_r
    diameter_inner_r=roller_screw_temp(1,54);
    %%%记录内啮合的有效螺纹长度(mm)，length_mesh_inner
    length_mesh_inner=roller_screw_temp(1,58);
    %%%记录外啮合的有效螺纹长度(mm)，length_mesh_outer
    length_mesh_outer=roller_screw_temp(1,59);
    %%%记录丝杠弹性模量(Mpa)，E_s
    E_s=roller_screw_temp(1,74);
    %%%记录螺母弹性模量(Mpa)，E_n
    E_n=roller_screw_temp(1,75);
    %%%记录滚柱弹性模量(Mpa)，E_r
    E_r=roller_screw_temp(1,76);
    %%%记录丝杠泊松比，possion_s
    possion_s=roller_screw_temp(1,77);
    %%%记录螺母泊松比，possion_n
    possion_n=roller_screw_temp(1,78);
    %%%记录滚柱泊松比，possion_r
    possion_r=roller_screw_temp(1,79);
    %%%计算丝杠综合模量，E0_s
    E0_s=E_s/(1-possion_s^2);
    %%%计算螺母综合模量，E0_n
    E0_n=E_n/(1-possion_n^2);
    %%%计算滚柱综合模量，E0_r
    E0_r=E_r/(1-possion_r^2);
    %%%计算螺母截面积(mm^2)，area_n
    area_n=pi*(diameter_outer_n^2-D_bottom_n^2)/4;
    %%%计算丝杠截面积(mm^2)，area_s
    area_s=pi*(D_bottom_s^2-diameter_inner_s^2)/4;
    %%%计算滚柱截面积(mm^2)，area_r
    area_r=pi*(D_bottom_r^2-diameter_inner_r^2)/4;
    %%%
    %%%记录行星滚柱丝杠的序号为order_load个工况的丝杠工作面，face_screw
    %行星滚柱丝杠外螺纹/内螺纹的法向载荷相关矩阵，norm_roller_screw，[1 工况序号 2 行星滚柱丝杠序号 3 丝杠功率(kW) 4 螺母功率(kW) 5  无预紧效率
    %6 丝杠轴序号 7 丝杠工作面(-1表示下轮廓，1表示上轮廓) 8 丝杠工作压力角(rad) 9 丝杠受载个数 10 丝杠法向载荷(N) 11 丝杠的法向矢量X轴分量 12 丝杠的法向矢量Y轴分量 13 丝杠的法向矢量Z轴分量
    %14 螺母轴序号 15 螺母工作面(-1表示下轮廓，1表示上轮廓) 16 螺母工作压力角(rad) 17 螺母受载个数 18 螺母法向载荷(N) 19 螺母的法向矢量X轴分量 20 螺母的法向矢量Y轴分量 21 螺母的法向矢量Z轴分量
    %22 旋转机构因预紧力的法向载荷 23 驱动形式(1表示正向驱动，2表示逆向驱动) 24 直线机构线速度(mm/s),velocity_linear 25 旋转机构自转速度(rpm),speed_turn 26 滚柱相对转动速度(rpm),speed_roller 27 持续时间(s),time 28 循环次数，number_cycle 29 行星滚柱丝杠的类型(1表示标准式，2表示反向式),type_roller_screw]
    id_norm_1=(norm_roller_screw(:,1)==order_load); %序号为order_load个工况序号，order_load
    id_norm_2=(norm_roller_screw(:,2)==order_roller_screw); %行星滚柱丝杠的序号，order_roller_screw
    id_norm=(id_norm_1+id_norm_2==2);
    face_screw=norm_roller_screw(id_norm,7);
    %%%记录丝杠工作压力角(rad)，press_angle_screw
    press_angle_screw=abs(norm_roller_screw(id_norm,8));
    %%%记录丝杠法向载荷(N)，force_screw
    force_screw=abs(norm_roller_screw(id_norm,10));
    %%%记录丝杠的受载法向矢量，vector_norm_screw
    vector_norm_screw=norm_roller_screw(id_norm,11:13);
    %%%计算丝杠轴向分量比例，ratio_screw
    ratio_screw=abs(vector_norm_screw(1,3))/norm(vector_norm_screw);
    %%%记录行星滚柱丝杠的序号为order_load个工况的螺母工作面，face_nut
    face_nut=norm_roller_screw(id_norm,15);
    %%%记录螺母工作压力角(rad)，press_angle_nut
    press_angle_nut=abs(norm_roller_screw(id_norm,16));
    %%%记录螺母法向载荷(N)，force_nut
    force_nut=abs(norm_roller_screw(id_norm,18));
    %%%记录螺母的受载法向矢量，vector_norm_nut
    vector_norm_nut=norm_roller_screw(id_norm,19:21);
    %%%计算螺母轴向分量比例，ratio_nut
    ratio_nut=abs(vector_norm_nut(1,3))/norm(vector_norm_nut);
    %%%
    %%%记录丝杠外啮合点半径(mm)，radius_screw
    %行星滚柱丝杠螺纹外啮点位置矩阵，mesh_outer_matrix，[1 行星滚柱丝杠序号 2 丝杠工作面(-1表示下轮廓，1表示上轮廓) 3 滚柱工作面(-1表示下轮廓，1表示上轮廓) 4 外啮合点全局坐标系X轴坐标(mm) 5 外啮合点全局坐标系Y轴坐标(mm) 6 丝杠外啮合点夹角(rad) 7 丝杠外啮合点半径(mm) 8 滚柱外啮合夹角(rad) 9 外啮合点半径(mm) 10 外啮点轴向间隙(mm)]
    id_mesh_outer_1=(mesh_outer_matrix(:,1)==order_roller_screw); %行星滚柱丝杠的序号，order_roller_screw
    id_mesh_outer_2=(mesh_outer_matrix(:,2)==face_screw); %丝杠工作面，face_screw
    id_mesh_outer=(id_mesh_outer_1+id_mesh_outer_2==2);
    radius_screw=mesh_outer_matrix(id_mesh_outer,7);
    %%%记录滚柱外啮合半径(mm)，radius_outer
    radius_outer=mesh_outer_matrix(id_mesh_outer,9);
    %%%
    %%%记录螺母内啮合点半径(mm)，radius_nut
    %行星滚柱丝杠螺纹内啮点位置矩阵，mesh_inner_matrix，[1 行星滚柱丝杠序号 2 螺母工作面(-1表示下轮廓，1表示上轮廓) 3 滚柱工作面(-1表示下轮廓，1表示上轮廓) 4 内啮合点全局坐标系X轴坐标(mm) 5 内啮合点全局坐标系Y轴坐标(mm) 6 螺母内啮合点夹角(rad) 7 螺母内啮合点半径(mm) 8 滚柱内啮合夹角(rad) 9 内啮合点半径(mm) 10 内啮点轴向间隙(mm)]
    id_mesh_inner_1=(mesh_inner_matrix(:,1)==order_roller_screw); %行星滚柱丝杠的序号，order_roller_screw
    id_mesh_inner_2=(mesh_inner_matrix(:,2)==face_nut); %螺母工作面，face_nut
    id_mesh_inner=(id_mesh_inner_1+id_mesh_inner_2==2);
    radius_nut=mesh_inner_matrix(id_mesh_inner,7);
    %%%记录滚柱内啮合半径(mm)，radius_inner
    radius_inner=mesh_inner_matrix(id_mesh_inner,9);
    %%%
    %%%计算外啮合滚柱第1主曲率(1/mm)，p_outer_1
    %滚柱牙型圆弧半径(mm)，radius_roller
    p_outer_1=1/radius_roller;
    %%%计算外啮合滚柱第2主曲率(1/mm),p_outer_2
    %滚柱外啮合半径(mm)，radius_outer
    %丝杠工作压力角(rad)，press_angle_screw
    p_outer_2=cos(pi/2-press_angle_screw)/radius_outer;
    %%%计算外啮合丝杠第1主曲率(1/mm),p_screw_1
    p_screw_1=0;
    %%%计算外啮合丝杠第2主曲率(1/mm)，p_screw_2
    %丝杠工作压力角(rad)，press_angle_screw
    p_screw_2=cos(pi/2-press_angle_screw)/radius_screw;
    %%%计算外啮合曲率之和(1/mm)，sum_p_outer
    sum_p_outer=p_outer_1+p_outer_2+p_screw_1+p_screw_2;
    %%%计算外啮合主曲率函数，F_p_outer
    F_p_outer=(abs(p_outer_1-p_outer_2)+abs(p_screw_1-p_screw_2))/sum_p_outer;
    %%%计算外啮合综合模量影响系数，E0_outer
    E0_outer=(1.137*10^5*(1/E0_s+1/E0_r))^(1/3);
    %%%计算外啮合压缩势能辅助值，B_outer
    B_outer=0.25*(p_outer_1+p_outer_2)+(p_screw_1+p_screw_2)-abs(p_outer_1-p_outer_2)-abs(p_screw_1-p_screw_2);
    %%%
    %%%计算内啮合滚柱第1主曲率(1/mm)，p_inner_1
    %滚柱牙型圆弧半径(mm)，radius_roller
    p_inner_1=1/radius_roller;
    %%%计算内啮合滚柱第2主曲率(1/mm),p_inner_2
    %滚柱内啮合半径(mm)，radius_inner
    %螺母工作压力角(rad)，press_angle_nut
    p_inner_2=cos(pi/2-press_angle_nut)/radius_inner;
    %%%计算内啮合螺母第1主曲率(1/mm),p_nut_1
    p_nut_1=0;
    %%%计算内啮合螺母第2主曲率(1/mm)，p_nut_2
    %螺母工作压力角(rad)，press_angle_nut
    p_nut_2=-1*cos(pi/2-press_angle_nut)/radius_nut;
    %%%计算内啮合曲率之和(1/mm)，sum_p_inner
    sum_p_inner=p_inner_1+p_inner_2+p_nut_1+p_nut_2;
    %%%计算内啮合主曲率函数，F_p_inner
    F_p_inner=(abs(p_inner_1-p_inner_2)+abs(p_nut_1-p_nut_2))/sum_p_inner;
    %%%计算内啮合综合模量影响系数，E0_inner
    E0_inner=(1.137*10^5*(1/E0_n+1/E0_r))^(1/3);
    %%%计算内啮合压缩势能辅助值，B_inner
    B_inner=0.25*(p_inner_1+p_inner_2)+(p_nut_1+p_nut_2)-abs(p_inner_1-p_inner_2)-abs(p_nut_1-p_nut_2);
    %%%
    %%%计算外啮合的长半轴/短半轴/接触变形影响系数，C_a_outer/C_b_outer/C_delta_outer
    %赫兹接触系数表，H
    C_a_outer=interp1(H(:,1),H(:,2),F_p_outer,'spline','extrap');
    C_b_outer=interp1(H(:,1),H(:,3),F_p_outer,'spline','extrap');
    C_delta_outer=interp1(H(:,1),H(:,4),F_p_outer,'spline','extrap');
    %%%
    %%%计算内啮合的长半轴/短半轴/接触变形影响系数，C_a_inner/C_b_inner/C_delta_inner
    %赫兹接触系数表，H
    C_a_inner=interp1(H(:,1),H(:,2),F_p_inner,'spline','extrap');
    C_b_inner=interp1(H(:,1),H(:,3),F_p_inner,'spline','extrap');
    C_delta_inner=interp1(H(:,1),H(:,4),F_p_inner,'spline','extrap');
    %%%
    %%%计算理论受载螺纹牙数，number
    number=1+fix(min(length_mesh_inner,length_mesh_outer)/pitch_screw);
    %%%判断螺母侧修形向量维数是否正确
    [row_modify,column_modify]=size(modify_nut);
    if (row_modify~=number)||(column_modify~=number_roller)
        h1=msgbox('螺母侧修形向量维数不正确，程序中止');
        javaFrame=get(h1,'JavaFrame');
        javaFrame.setFigureIcon(javax.swing.ImageIcon('飞机.jpg'));
        %%%退出程序
        return;
    end
    %%%判断丝杠侧修形向量维数是否正确
    [row_modify,column_modify]=size(modify_screw);
    if (row_modify~=number)||(column_modify~=number_roller)
        h1=msgbox('丝杠侧修形向量维数不正确，程序中止');
        javaFrame=get(h1,'JavaFrame');
        javaFrame.setFigureIcon(javax.swing.ImageIcon('飞机.jpg'));
        %%%退出程序
        return;
    end
    %%%计算载荷作用点，number_load
    number_load=number*number_roller;
    %%%计算行星滚柱丝杠总的轴向载荷(N)，force_axial
    if type_roller_screw==1 %行星滚柱丝杠的(1表示标准式，2表示反向式),type_roller_screw
        %螺母法向载荷(N)，force_nut
        %螺母轴向分量比例，ratio_nut
        force_axial=abs(force_nut)*ratio_nut*number_load;
    elseif type_roller_screw==2 %行星滚柱丝杠的(1表示标准式，2表示反向式),type_roller_screw
        %丝杠法向载荷(N)，force_screw
        %%丝杠轴向分量比例，ratio_screw
        force_axial=abs(force_screw)*ratio_screw*number_load;
    end
    %%%
    %%%计算行星滚柱丝杠内螺纹/外螺纹的啮合柔度
    [flexible_inner,flexible_outer]=flexible_roller_screw(roller_screw_temp);
    %内螺纹的啮合柔度(mm/N)，flexible_inner
    %外螺纹的啮合柔度(mm/N)，flexible_outer
    %%%
    %%%求解行星滚柱丝杠各接触点的受载法向载荷
    if type_load==1 %行星滚柱丝杠受载方式，type_load，1表示同侧受载，2表示异侧受载
        %%%设置初始值
        F0=force_nut*ones(2*number,number_roller);
        %%%设置螺母侧零工作侧隙(mm)，nut_temp
        nut_temp=zeros(number,number_roller);
        %%%设置丝杠侧零工作侧隙(mm)，screw_temp
        screw_temp=zeros(number,number_roller);
        %%%定义螺母侧工作侧隙极大值向量(mm)，max_nut
        max_nut=zeros(number,number_roller);
        %%%定义丝杠侧工作侧隙极大值向量(mm)，max_screw
        max_screw=zeros(number,number_roller);
        %%%定义滚柱载荷分配系数，share
% % %         share=[0.16;0.21;0.21;0.21;0.21];
        share=(1/number_roller)*ones(number_roller,1);
        %%%调用fsolve求解零工作侧隙的法向载荷，Fn_refer
        options=optimoptions('fsolve','OptimalityTolerance',10^-6);
        [Fn_refer]=fsolve(@(Fn)load_roller_screw_1(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,nut_temp,screw_temp,share),F0,options);
        %%%计算考虑实际工作侧隙的各接触点法向载荷
        if (sum(sum(modify_nut))+sum(sum(modify_screw)))>0
            %%%计算螺母侧/丝杠侧工作侧隙极值(mm)，max_nut/max_screw
            for kk=1:number_roller %行星滚柱丝杠的滚柱个数，number_roller
                for ii=2:number %理论受载螺纹牙数，number
                    Fn_temp=Fn_refer;
                    Fn_temp(ii-1,kk)=0;
                    Fn_temp(ii-1+number,kk)=0;
                    max_nut(ii-1,kk)=sum(sum(max(0,Fn_temp(ii:number,1:number_roller))))*ratio_nut*pitch_screw/(area_n*E_n)-C_delta_inner*E0_inner^2*sum_p_inner^(1/3)*(max(0,Fn_temp(ii-1,kk))^(2/3)-max(0,Fn_temp(ii,kk))^(2/3))/ratio_nut-flexible_inner*(max(0,Fn_temp(ii-1,kk))-max(0,Fn_temp(ii,kk)))+(sum(max(0,Fn_temp(ii:number,kk)))*ratio_nut-sum(max(0,Fn_temp(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r);
                    max_screw(ii-1,kk)=sum(sum(max(0,Fn_temp(ii+number:2*number,1:number_roller))))*ratio_screw*pitch_screw/(area_s*E_s)-C_delta_outer*E0_outer^2*sum_p_outer^(1/3)*(max(0,Fn_temp(ii-1+number,kk))^(2/3)-max(0,Fn_temp(ii+number,kk))^(2/3))/ratio_screw-flexible_outer*(max(0,Fn_temp(ii-1+number,kk))-max(0,Fn_temp(ii+number,kk)))-(sum(max(0,Fn_temp(ii:number,kk)))*ratio_nut-sum(max(0,Fn_temp(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r);
                end
                max_nut(ii,kk)=max_nut(ii-1,kk);
                max_screw(ii,kk)=max_screw(ii-1,kk);
            end
            max_nut=0.90*max_nut;
            max_screw=0.90*max_screw;
            %%%调整螺母侧/丝杠侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_nut/modify_screw
            modify_nut=min(modify_nut,max_nut);
            modify_screw=min(modify_screw,max_screw);
            %%%求解考虑工作侧隙时行星滚柱丝杠各接触点的受载法向载荷
            [Fn,fval]=fsolve(@(Fn)load_roller_screw_1(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share),F0,options);
            if (sum(sum(abs(fval(2:number,1:number_roller)))>0.01)>0)||(sum(sum(abs(fval(2+number:2*number,1:number_roller)))>0.01)>0)
                h1=msgbox('行星滚柱丝杠法向载荷计算存在不收敛');
                javaFrame=get(h1,'JavaFrame');
                javaFrame.setFigureIcon(javax.swing.ImageIcon('飞机.jpg'));
            end
        else
            %%%定义实际工作侧隙的各接触点法向载荷
            Fn=Fn_refer;
        end
        %%%调整行星滚柱丝杠各接触点的受载法向载荷
        Fn(:,:)=max(0,Fn(:,:));
        %%%记录螺母的接触点法向载荷的向量，Fn_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
        Fn_nut(1:number,1)=(1:number)';
        Fn_nut(1:number,2:1+number_roller)=Fn(1:number,1:number_roller);
        %%%记录螺母的接触点法向载荷不均载系数的向量，distribution_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
        distribution_nut(1:number,1)=(1:number)';
        distribution_nut(1:number,2:1+number_roller)=Fn_nut(1:number,2:1+number_roller)/(force_axial/number_load/ratio_nut);
        %%%丝杠的接触点法向载荷的向量，Fn_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
        Fn_screw(1:number,1)=(1:number)';
        Fn_screw(1:number,2:1+number_roller)=Fn(1+number:2*number,1:number_roller);
        %%%记录螺母的接触点法向载荷不均载系数的向量，distribution_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
        distribution_screw(1:number,1)=(1:number)';
        distribution_screw(1:number,2:1+number_roller)=Fn_screw(1:number,2:1+number_roller)/(force_axial/number_load/ratio_screw);
    end
    %%%
    if type_load==2 %行星滚柱丝杠受载方式，type_load，1表示同侧受载，2表示异侧受载
        %%%设置初始值
        F0=force_nut*ones(2*number,number_roller);
        %%%设置螺母侧零工作侧隙(mm)，nut_temp
        nut_temp=zeros(number,number_roller);
        %%%设置丝杠侧零工作侧隙(mm)，screw_temp
        screw_temp=zeros(number,number_roller);
        %%%定义螺母侧工作侧隙极大值向量(mm)，max_nut
        max_nut=zeros(number,number_roller);
        %%%定义丝杠侧工作侧隙极大值向量(mm)，max_screw
        max_screw=zeros(number,number_roller);
        %%%定义滚柱载荷分配系数，share
% % %         share=[0.16;0.21;0.21;0.21;0.21];
        share=(1/number_roller)*ones(number_roller,1);
        %%%调用fsolve求解零工作侧隙的法向载荷，Fn_refer
        options=optimoptions('fsolve','OptimalityTolerance',10^-6);
        [Fn_refer]=fsolve(@(Fn)load_roller_screw_2(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,nut_temp,screw_temp,share),F0,options);
        %%%计算考虑实际工作侧隙的各接触点法向载荷
        if (sum(sum(modify_nut))+sum(sum(modify_screw)))>0
            %%%计算螺母侧/丝杠侧工作侧隙极值(mm)，max_nut/max_screw
            for kk=1:number_roller %行星滚柱丝杠的滚柱个数，number_roller
                for ii=2:number %理论受载螺纹牙数，number
                    Fn_temp=Fn_refer;
                    Fn_temp(ii-1,kk)=0;
                    Fn_temp(ii+number,kk)=0;
                    max_nut(ii-1,kk)=sum(sum(max(0,Fn_temp(ii:number,1:number_roller))))*ratio_nut*pitch_screw/(area_n*E_n)-C_delta_inner*E0_inner^2*sum_p_inner^(1/3)*(max(0,Fn_temp(ii-1,kk))^(2/3)-max(0,Fn_temp(ii,kk))^(2/3))/ratio_nut-flexible_inner*(max(0,Fn_temp(ii-1,kk))-max(0,Fn_temp(ii,kk)))+(sum(max(0,Fn_temp(ii:number,kk)))*ratio_nut-sum(max(0,Fn_temp(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r);
                    max_screw(ii,kk)=sum(sum(max(0,Fn_temp(1+number:ii-1+number,1:number_roller))))*ratio_screw*pitch_screw/(area_s*E_s)-C_delta_outer*E0_outer^2*sum_p_outer^(1/3)*(max(0,Fn_temp(ii+number,kk))^(2/3)-max(0,Fn_temp(ii-1+number,kk))^(2/3))/ratio_screw-flexible_outer*(max(0,Fn_temp(ii+number,kk))-max(0,Fn_temp(ii-1+number,kk)))+(sum(max(0,Fn_temp(ii:number,kk)))*ratio_nut-sum(max(0,Fn_temp(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r);
                end
                max_nut(ii,kk)=max_nut(ii-1,kk);
                max_screw(1,kk)=max_screw(2,kk);
            end
            max_nut=0.90*max_nut;
            max_screw=0.90*max_screw;
            %%%调整螺母侧/丝杠侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_nut/modify_screw
            modify_nut=min(modify_nut,max_nut);
            modify_screw=min(modify_screw,max_screw);
            %%%求解考虑工作侧隙时行星滚柱丝杠各接触点的受载法向载荷
            [Fn,fval]=fsolve(@(Fn)load_roller_screw_2(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share),F0,options);
            if (sum(sum(abs(fval(2:number,1:number_roller)))>0.01)>0)||(sum(sum(abs(fval(2+number:2*number,1:number_roller)))>0.01)>0)
                h1=msgbox('行星滚柱丝杠法向载荷计算存在不收敛');
                javaFrame=get(h1,'JavaFrame');
                javaFrame.setFigureIcon(javax.swing.ImageIcon('飞机.jpg'));
            end
        else
            %%%定义实际工作侧隙的各接触点法向载荷
            Fn=Fn_refer;
        end
        %%%调整行星滚柱丝杠各接触点的受载法向载荷
        Fn(:,:)=max(0,Fn(:,:));
        %%%记录螺母/丝杠的接触点法向载荷的向量
        %%%记录螺母的接触点法向载荷的向量，Fn_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
        Fn_nut(1:number,1)=(1:number)';
        Fn_nut(1:number,2:1+number_roller)=Fn(1:number,1:number_roller);
        %%%记录螺母的接触点法向载荷不均载系数的向量，distribution_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
        distribution_nut(1:number,1)=(1:number)';
        distribution_nut(1:number,2:1+number_roller)=Fn_nut(1:number,2:1+number_roller)/(force_axial/number_load/ratio_nut);
        %%%丝杠的接触点法向载荷的向量，Fn_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
        Fn_screw(1:number,1)=(1:number)';
        Fn_screw(1:number,2:1+number_roller)=Fn(1+number:2*number,1:number_roller);
        %%%记录螺母的接触点法向载荷不均载系数的向量，distribution_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
        distribution_screw(1:number,1)=(1:number)';
        distribution_screw(1:number,2:1+number_roller)=Fn_screw(1:number,2:1+number_roller)/(force_axial/number_load/ratio_screw);
    end
    %%%
    %%%计算外啮合的接触长半轴(mm)，half_width_a_outer
    %外啮合综合模量影响系数，E0_outer
    %丝杠法向载荷(N)，force_screw
    %外啮合曲率之和(1/mm)，sum_p_outer
    half_width_a_outer(:,1)=(1:number)';
    half_width_a_outer(:,2:1+number_roller)=C_a_outer*E0_outer*(Fn_screw(:,2:1+number_roller)/sum_p_outer).^(1/3);
    %%%计算外啮合的接触短半轴(mm)，half_width_b_outer
    half_width_b_outer(:,1)=(1:number)';
    half_width_b_outer(:,2:1+number_roller)=C_b_outer*E0_outer*(Fn_screw(:,2:1+number_roller)/sum_p_outer).^(1/3);
    %%%计算外啮合接触应力(Mpa)，stress_outer
    stress_outer(:,1)=(1:number)';
    stress_outer(:,2:1+number_roller)=1.5*Fn_screw(:,2:1+number_roller)./(pi*half_width_a_outer(:,2:1+number_roller).*half_width_b_outer(:,2:1+number_roller));
    %%%计算外啮合接触变形量(mm)，delta_outer
    delta_outer(:,1)=(1:number)';
    delta_outer(:,2:1+number_roller)=C_delta_outer*(E0_outer^2)*(sum_p_outer*(Fn_screw(:,2:1+number_roller).^2)).^(1/3);
    %%%
    %%%计算内啮合的接触长半轴(mm)，half_width_a_inner
    %内啮合综合模量影响系数，E0_inner
    %丝杠法向载荷(N)，force_screw
    %内啮合曲率之和(1/mm)，sum_p_inner
    half_width_a_inner(:,1)=(1:number)';
    half_width_a_inner(:,2:1+number_roller)=C_a_inner*E0_inner*(Fn_nut(:,2:1+number_roller)/sum_p_inner).^(1/3);
    %%%计算内啮合的接触短半轴(mm)，half_width_b_inner
    half_width_b_inner(:,1)=(1:number)';
    half_width_b_inner(:,2:1+number_roller)=C_b_inner*E0_inner*(Fn_nut(:,2:1+number_roller)/sum_p_inner).^(1/3);
    %%%计算内啮合接触应力(Mpa)，stress_inner
    stress_inner(:,1)=(1:number)';
    stress_inner(:,2:1+number_roller)=1.5*Fn_nut(:,2:1+number_roller)./(pi*half_width_a_inner(:,2:1+number_roller).*half_width_b_inner(:,2:1+number_roller));
    %%%计算内啮合接触变形量(mm)，delta_inner
    delta_inner(:,1)=(1:number)';
    delta_inner(:,2:1+number_roller)=C_delta_inner*(E0_inner^2)*(sum_p_inner*(Fn_nut(:,2:1+number_roller).^2)).^(1/3);
    %%%
    %%%计算螺母载荷分布均方值，S_load_nut
    S_load_nut=(sum((Fn_nut(:,2:1+number_roller)-sum(Fn_nut(:,2:1+number_roller))/number).^2)/number).^(0.5);
    %%%计算丝杠载荷分布均方值，S_load_screw
    S_load_screw=(sum((Fn_screw(:,2:1+number_roller)-sum(Fn_screw(:,2:1+number_roller))/number).^2)/number).^(0.5);
else
    %%%定义螺母的接触点法向载荷的向量，Fn_nut，[1 接触点序号(注:序号从螺母受载端开始) 2 接触点法向载荷(N) 3 不均载系数]
    Fn_nut=zeros(1,2);
    %%%定义丝杠的接触点法向载荷的向量，Fn_screw，[1 接触点序号(注:序号从螺母受载端开始) 2 接触点法向载荷(N) 3 不均载系数]
    Fn_screw=zeros(1,2);
    %%%定义螺母的接触点法向载荷不均载系数的向量，distribution_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
    distribution_nut=zeros(1,2);
    %%%定义螺母的接触点法向载荷不均载系数的向量，distribution_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷的不均载系数]
    distribution_screw=zeros(1,2);
    %%%定义外啮合的接触长半轴(mm)，half_width_a_outer
    half_width_a_outer=zeros(1,1);
    %%%定义外啮合的接触短半轴(mm)，half_width_b_outer
    half_width_b_outer=zeros(1,1);
    %%%定义外啮合接触应力(Mpa)，stress_outer
    stress_outer=zeros(1,1);
    %%%定义外啮合接触变形量(mm)，delta_outer
    delta_outer=zeros(1,1);
    %%%定义内啮合的接触长半轴(mm)，half_width_a_inner
    half_width_a_inner=zeros(1,1);
    %%%定义内啮合的接触短半轴(mm)，half_width_b_inner
    half_width_b_inner=zeros(1,1);
    %%%定义内啮合接触应力(Mpa)，stress_inner
    stress_inner=zeros(1,1);
    %%%定义内啮合接触变形量(mm)，delta_inner
    delta_inner=zeros(1,1);
    %%%定义螺母载荷分布均方值，S_load_nut
    S_load_nut=zeros(1,1);
    %%%定义丝杠载荷分布均方值，S_load_screw
    S_load_screw=zeros(1,1);
end
%%%
%%%计算行星滚柱丝杠总的势能
[potential_total]=potential_roller_screw(order_roller_screw,roller_screw_initial,type_load,Fn_nut,Fn_screw,ratio_nut,ratio_screw,flexible_inner,flexible_outer,B_inner,B_outer,half_width_a_inner,half_width_b_inner,stress_inner,half_width_a_outer,half_width_b_outer,stress_outer);
%行星滚柱丝杠总的势能，potential_total
end
%%%主程序结束


%%%构建同侧受载行星滚柱丝杠的方程组(螺母受压，丝杠受拉)
function FF=load_roller_screw_1(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share)
%%%参数输出
%方程组右侧常数项，FF
%%%参数输入
%行星滚柱丝杠的法向载荷，注：第1列至第Z列 螺母法向载荷(N)；第Z+1列至第2*Z列 丝杠法向载荷(N)
%行星滚柱丝杠总的轴向载荷(N)，force_axial
%螺距(mm)，pitch_screw
%行星滚柱丝杠的滚柱个数，number_roller
%螺母轴向分量比例，ratio_nut
%丝杠轴向分量比例，ratio_screw
%螺母弹性模量(Mpa)，E_n
%丝杠弹性模量(Mpa)，E_s
%滚柱弹性模量(Mpa)，E_r
%螺母截面积(mm^2)，area_n
%丝杠截面积(mm^2)，area_s
%滚柱截面积(mm^2)，area_r
%内啮合的接触变形影响系数，C_delta_inner
%外啮合的接触变形影响系数，C_delta_outer
%内啮合综合模量影响系数，E0_inner
%外啮合综合模量影响系数，E0_outer
%内啮合曲率之和(1/mm)，sum_p_inner
%外啮合曲率之和(1/mm)，sum_p_outer
%内螺纹的啮合柔度(mm/N)，flexible_inner
%外螺纹的啮合柔度(mm/N)，flexible_outer
%螺母侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_nut
%丝杠侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_screw
%滚柱载荷分配系数，share
%%%程序开始
%%%计算法向载荷个数，number
number=length(Fn(:,1))/2;
%%%定义方程组右侧常数项，FF
FF=zeros(2*number,number_roller);
%%%依据螺母与滚柱的位移协调，构建螺母法向载荷方程组
for kk=1:number_roller %行星滚柱丝杠的滚柱个数，number_roller
    for ii=1:number %法向载荷个数，number
        if ii==1
            FF(ii,kk)=sum(max(0,Fn(1:number,kk)))-share(kk,1)*force_axial/ratio_nut;
        else
            FF(ii,kk)=10^3*(sum(sum(max(0,Fn(ii:number,1:number_roller))))*ratio_nut*pitch_screw/(area_n*E_n)-C_delta_inner*E0_inner^2*sum_p_inner^(1/3)*(max(0,Fn(ii-1,kk))^(2/3)-max(0,Fn(ii,kk))^(2/3))/ratio_nut-flexible_inner*(max(0,Fn(ii-1,kk))-max(0,Fn(ii,kk)))-(modify_nut(ii-1,kk)-modify_nut(ii,kk))+(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
    %%%
    %%%依据丝杠与滚柱的位移协调，构建丝杠法向载荷方程组
    for ii=1:number %法向载荷个数，number
        if ii==1
            FF(ii+number,kk)=sum(max(0,Fn(1+number:2*number,kk)))-share(kk,1)*force_axial/ratio_screw;
        else
            FF(ii+number,kk)=10^3*(sum(sum(max(0,Fn(ii+number:2*number,1:number_roller))))*ratio_screw*pitch_screw/(area_s*E_s)-C_delta_outer*E0_outer^2*sum_p_outer^(1/3)*(max(0,Fn(ii-1+number,kk))^(2/3)-max(0,Fn(ii+number,kk))^(2/3))/ratio_screw-flexible_outer*(max(0,Fn(ii-1+number,kk))-max(0,Fn(ii+number,kk)))-(modify_screw(ii-1,kk)-modify_screw(ii,kk))-(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
end
end
%%%程序结束

%%%构建异侧受载行星滚柱丝杠的方程组(螺母受压，丝杠受压)
function FF=load_roller_screw_2(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share)
%%%参数输出
%方程组右侧常数项，FF
%%%参数输入
%行星滚柱丝杠的法向载荷，注：第1列至第Z列 螺母法向载荷(N)；第Z+1列至第2*Z列 丝杠法向载荷(N)
%行星滚柱丝杠总的轴向载荷(N)，force_axial
%螺距(mm)，pitch_screw
%行星滚柱丝杠的滚柱个数，number_roller
%螺母轴向分量比例，ratio_nut
%丝杠轴向分量比例，ratio_screw
%螺母弹性模量(Mpa)，E_n
%丝杠弹性模量(Mpa)，E_s
%滚柱弹性模量(Mpa)，E_r
%螺母截面积(mm^2)，area_n
%丝杠截面积(mm^2)，area_s
%滚柱截面积(mm^2)，area_r
%内啮合的接触变形影响系数，C_delta_inner
%外啮合的接触变形影响系数，C_delta_outer
%内啮合综合模量影响系数，E0_inner
%外啮合综合模量影响系数，E0_outer
%内啮合曲率之和(1/mm)，sum_p_inner
%外啮合曲率之和(1/mm)，sum_p_outer
%内螺纹的啮合柔度(mm/N)，flexible_inner
%外螺纹的啮合柔度(mm/N)，flexible_outer
%螺母侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_nut
%丝杠侧修形向量(注：螺母加载端为序号开始端，且工作半侧隙)(mm)，modify_screw
%滚柱载荷分配系数，share
%%%程序开始
%%%计算法向载荷个数，number
number=length(Fn(:,1))/2;
%%%定义方程组右侧常数项，FF
FF=zeros(2*number,number_roller);
%%%依据螺母与滚柱的位移协调，构建螺母法向载荷方程组
for kk=1:number_roller %行星滚柱丝杠的滚柱个数，number_roller
    for ii=1:number %法向载荷个数，number
        if ii==1
            FF(ii,kk)=sum(max(0,Fn(1:number,kk)))-share(kk,1)*force_axial/ratio_nut;
        else
            FF(ii,kk)=10^3*(sum(sum(max(0,Fn(ii:number,1:number_roller))))*ratio_nut*pitch_screw/(area_n*E_n)-C_delta_inner*E0_inner^2*sum_p_inner^(1/3)*(max(0,Fn(ii-1,kk))^(2/3)-max(0,Fn(ii,kk))^(2/3))/ratio_nut-flexible_inner*(max(0,Fn(ii-1,kk))-max(0,Fn(ii,kk)))-(modify_nut(ii-1,kk)-modify_nut(ii,kk))+(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
    %%%
    %%%依据丝杠与滚柱的位移协调，构建丝杠法向载荷方程组
    for ii=1:number %法向载荷个数，number
        if ii==1
            FF(ii+number,kk)=sum(max(0,Fn(1+number:2*number,kk)))-share(kk,1)*force_axial/ratio_screw;
        else
            FF(ii+number,kk)=10^3*(sum(sum(max(0,Fn(1+number:ii-1+number,1:number_roller))))*ratio_screw*pitch_screw/(area_s*E_s)-C_delta_outer*E0_outer^2*sum_p_outer^(1/3)*(max(0,Fn(ii+number,kk))^(2/3)-max(0,Fn(ii-1+number,kk))^(2/3))/ratio_screw-flexible_outer*(max(0,Fn(ii+number,kk))-max(0,Fn(ii-1+number,kk)))-(modify_screw(ii,kk)-modify_screw(ii-1,kk))+(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
end
end
%%%程序结束

%%%计算行星滚柱丝杠内螺纹/外螺纹的啮合柔度
function [flexible_inner,flexible_outer]=flexible_roller_screw(roller_screw_temp)
%%%参数输出
%内螺纹的啮合柔度(mm/N)，flexible_inner
%外螺纹的啮合柔度(mm/N)，flexible_outer
%%%参数输入
%行星滚柱丝杠基本参数，roller_screw_temp，[1 行星滚柱丝杠序号 2 行星滚柱丝杠的类型(1表示标准式，2表示反向式),type_roller_screw 3 滚柱个数,number_roller 4 丝杠螺纹头数,N_thread_s 5 螺母螺纹头数,N_thread_n 6 滚柱螺纹头数,N_thread_r
%7 丝杠螺纹旋向(-1表示左旋，1表示右旋),direct_screw 8 螺母螺纹旋向(-1表示左旋，1表示右旋),direct_nut 9 滚柱螺纹旋向(-1表示左旋，1表示右旋),direct_roller 10 丝杠螺纹升角(rad)，rise_angle_s 11 螺母螺纹升角(rad)，rise_angle_n 12 滚柱螺纹升角(rad)，rise_angle_r
%13 第1个滚柱与X轴夹角(rad)，angle_roller 14 内螺纹当量摩擦系数，f_inner 15 外螺纹当量摩擦系数，f_outer 16 螺距(mm)，pitch_screw 17 牙型角(rad)，angle_screw 18 滚柱牙型圆弧半径(mm)，radius_roller 19 滚柱齿廓圆心截面坐标系X轴值(mm)，X_center_r 20 滚柱齿廓圆心截面坐标系Y轴值(mm)，Y_center_r
%21 丝杠螺纹削顶系数，cutter_top_s 22 丝杠螺纹削根系数，cutter_bottom_s 23 螺母螺纹削顶系数，cutter_top_n 24 螺母螺纹削根系数，cutter_bottom_n 25 滚柱螺纹削顶系数，cutter_top_r 26 滚柱螺纹削根系数，cutter_bottom_r 27 丝杠螺纹半牙减薄量(mm)，reduce_s 28 螺母螺纹半牙减薄量(mm)，reduce_n 29 滚柱螺纹半牙减薄量(mm)，reduce_r 30 空
%31 丝杠的螺纹齿顶高(mm)，addendum_s 32 丝杠的螺纹齿根高(mm)，dedendum_s 33 螺母的螺纹齿顶高(mm)，addendum_n 34 螺母的螺纹齿根高(mm)，dedendum_n 35 滚柱的螺纹齿顶高(mm)，addendum_r 36 滚柱的螺纹齿根高(mm)，dedendum_r
%37 丝杠实际顶径(mm)，D_top_s 38 丝杠实际根径(mm)，D_bottom_s 39 螺母实际顶径(mm)，D_top_n
%40 螺母实际根径(mm)，D_bottom_n 41 滚柱实际顶径(mm)，D_top_r 42 滚柱实际根径(mm)，D_bottom_r
%43 丝杠中径(mm),D_pitch_s ,44 螺母中径(mm),D_pitch_n, 45 滚柱中径(mm)，D_pitch_r
%46 丝杠轴长度(mm)，length_shaft_s 47 螺母轴长度(mm)，length_shaft_n 48 滚柱轴长度(mm)，length_shaft_r 49 丝杠轴外径(mm)，diameter_outer_s 50 螺母轴外径(mm)，diameter_outer_n 51 滚柱轴外径(mm)，diameter_outer_r 52 丝杠轴内径(mm)，diameter_inner_s 53 螺母轴内径(mm)，diameter_inner_n 54 滚柱轴内径(mm)，diameter_inner_r
%55 丝杠螺纹长度(mm)，length_thread_s 56 螺母螺纹长度(mm)，length_thread_n 57 滚柱螺纹长度(mm)，length_thread_r 58 内啮合的有效螺纹长度(mm)，length_mesh_inner 59 外啮合的有效螺纹长度(mm)，length_mesh_outer 60 空
%61 左齿轮宽度(mm)，width_gear_left 62 右齿轮宽度(mm)，width_gear_left 63 左保持架宽度(mm)，width_carrier_left 64 左保持架宽度(mm)，width_carrier_right
%65 螺母轴左端相对于丝杠轴左端的位置(mm)，delta_n_s 66 滚柱轴左端相对于丝杠轴左端的位置(mm)，delta_r_s 67 丝杠螺纹左端相对于丝杠轴左端的位置(mm),delta_thread_s 68 螺母螺纹左端相对于螺母轴左端的位置(mm),delta_thread_n 69 滚柱螺纹左端相对于滚柱轴左端的位置(mm),delta_thread_r 70 轴向预紧载荷(N)，preload_axial
%71 丝杠轴的密度(t/mm^-9)，density_s 72 螺母轴的密度(t/mm^-9)，density_n 73 滚柱轴的密度(t/mm^-9)，density_r 74 丝杠弹性模量(Mpa)，E_s 75 螺母弹性模量(Mpa)，E_n 76 滚柱弹性模量(Mpa)，E_r 77 丝杠泊松比，possion_s 78 螺母泊松比，possion_n 79 滚柱泊松比，possion_r
%80 左齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_left 81 右齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_right 82 左行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_left 83 右行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_right
%84 左保持架外径(mm)，outer_carrier_left 85 左保持架内径(mm)，inner_carrier_left 86 右保持架外径(mm)，outer_carrier_right 87 右保持架内径(mm)，inner_carrier_right 88 行星滚柱丝是否指定效率(0表示不指定，1表示指定),sign_efficiency 89 正向驱动效率(无预紧载荷)，advance_efficiency 90 逆向驱动效率(无预紧载荷)，reverse_efficiency
%91 齿轮法向模数(mm)，m_n 92 内齿圈齿数，z_r 93 太阳轮齿数，z_s 94 行星轮齿数，z_p 95 压力角(rad)，press_angle 96 端面压力角(rad)，tran_press_angle 97 螺旋角(rad)，helix_angle 98 内齿圈法向变位系数，n_r 99 太阳轮法向变位系数，n_s 100 行星轮法向变位系数，n_p
%101 工作中心距(mm)，work_center 102 行星轮系内啮合角，mesh_angle(rad) 103 行星轮系外啮合角，mesh_angle(rad) 104 内齿圈齿顶圆，tran_ad_dia_r（mm） 105 太阳轮齿顶圆，tran_ad_dia_s（mm） 106 行星轮齿顶圆，tran_ad_dia_p（mm） 107 内齿圈齿根圆，tran_de_dia_r（mm） 108 太阳轮齿根圆，tran_de_dia_s（mm） 109 行星轮齿根圆，tran_de_dia_p（mm）]
%110 内齿圈的节圆（mm），tran_pitch_dia_r 111 太阳轮的节圆（mm），tran_pitch_dia_s 112 行星轮的节圆（mm），tran_pitch_dia_p 113-120 空
%121 螺母幅板宽度(mm)，width_web_n 122 螺母幅板中心距螺母轴左端距离(mm)，delta_web_n 123 丝杠左幅板宽度(mm)，width_left_web_s 124 丝杠左幅板中心距丝杠轴左端距离(mm)，delta_left_web_s 125 丝杠右幅板宽度(mm)，width_right_web_s 126 丝杠右幅板中心距丝杠轴左端距离(mm)，delta_right_web_s 127-130 空
%131 螺母轮缘轴序号,shaft_self_nut 132 丝杠轮缘轴序号，shaft_self_screw 133 滚柱轮缘轴序号，shaft_self_roller,134-140 空
%141 行星轮刀具齿顶高系数，142 行星轮刀具顶隙系数，143 行星轮齿顶削减量，144 内齿圈刀具齿顶高系数，145 内齿圈刀具顶隙系数，146 内齿圈齿顶削减量,147 太阳轮刀具齿顶高系数，148 太阳轮刀具顶隙系数，149 太阳轮齿顶削减量
%150-160 空 161 有效内啮合螺纹相对于螺母轴左端的位置(mm)，thread_inner_nut 162 有效内啮合螺纹相对于滚柱轴左端的位置(mm)，thread_inner_roller 163 有效外啮合螺纹相对于丝杠轴左端的位置(mm)，thread_outer_screw 164 有效外啮合螺纹相对于滚柱轴左端的位置(mm)，thread_outer_roller]
%%%程序开始
%%%记录螺距(mm)，pitch_screw
pitch_screw=roller_screw_temp(1,16);
%%%记录牙型角(rad)，angle_screw
angle_screw=roller_screw_temp(1,17);
%%%记录丝杠螺纹削根系数，cutter_bottom_s
cutter_bottom_s=roller_screw_temp(1,22);
%%%记录螺母螺纹削根系数，cutter_bottom_n
cutter_bottom_n=roller_screw_temp(1,24);
%%%记录滚柱螺纹削根系数，cutter_bottom_r
cutter_bottom_r=roller_screw_temp(1,26);
%%%记录丝杠螺纹半牙减薄量(mm)，reduce_s
reduce_s=roller_screw_temp(1,27);
%%%记录螺母螺纹半牙减薄量(mm)，reduce_n
reduce_n=roller_screw_temp(1,28);
%%%记录滚柱螺纹半牙减薄量(mm)，reduce_r
reduce_r=roller_screw_temp(1,29);
%%%记录丝杠中径(mm)，D_pitch_s
D_pitch_s=roller_screw_temp(1,43);
%%%记录螺母中径(mm),D_pitch_n
D_pitch_n=roller_screw_temp(1,44);
%%%记录滚柱中径(mm)，D_pitch_r
D_pitch_r=roller_screw_temp(1,45);
%%%螺母轴外径(mm)，diameter_outer_n
diameter_outer_n=roller_screw_temp(1,50);
%%%记录丝杠弹性模量(Mpa)，E_s
E_s=roller_screw_temp(1,74);
%%%记录螺母弹性模量(Mpa)，E_n
E_n=roller_screw_temp(1,75);
%%%记录滚柱弹性模量(Mpa)，E_r
E_r=roller_screw_temp(1,76);
%%%记录丝杠泊松比，possion_s
possion_s=roller_screw_temp(1,77);
%%%记录螺母泊松比，possion_n
possion_n=roller_screw_temp(1,78);
%%%记录滚柱泊松比，possion_r
possion_r=roller_screw_temp(1,79);
%%%
%%%计算螺纹高度，high_thread
high_thread=(0.5*pitch_screw)/tan(0.5*angle_screw);
%%%计算螺母中径牙厚(mm)，b_nut
b_nut=0.5*pitch_screw-2*reduce_n;
%%%计算螺母底径牙厚(mm)，a_nut
a_nut=pitch_screw*(1-cutter_bottom_n)-2*reduce_n;
%%%计算螺母牙根高(mm)，c_nut
c_nut=(0.5-cutter_bottom_n)*high_thread;
%%%计算滚柱中径牙厚(mm)，b_roller
b_roller=0.5*pitch_screw-2*reduce_r;
%%%计算滚柱底径牙厚(mm)，a_roller
a_roller=pitch_screw*(1-cutter_bottom_r)-2*reduce_r;
%%%计算滚柱牙根高(mm)，c_roller
c_roller=(0.5-cutter_bottom_r)*high_thread;
%%%计算丝杠中径牙厚(mm)，b_screw
b_screw=0.5*pitch_screw-2*reduce_s;
%%%计算丝杠底径牙厚(mm)，a_screw
a_screw=pitch_screw*(1-cutter_bottom_s)-2*reduce_s;
%%%计算丝杠牙根高(mm)，c_screw
c_screw=(0.5-cutter_bottom_s)*high_thread;
%%%
%%%计算螺母啮合柔度(mm/N)，flexible_nut
flexible_nut_1=(1-possion_n^2)*(3/(4*E_n))*((1-(2-b_nut/a_nut)^2+2*log(a_nut/b_nut))*cot(0.5*angle_screw)^3-4*(c_nut/a_nut)^2*tan(0.5*angle_screw));
flexible_nut_2=(1+possion_n)*(6/(5*E_n))*cot(0.5*angle_screw)*log(a_nut/b_nut);
flexible_nut_3=(1-possion_n^2)*(12*c_nut/(pi*E_n*a_nut^2))*(c_nut-0.5*b_nut*tan(0.5*angle_screw));
flexible_nut_4=(1-possion_n^2)*(2/(pi*E_n))*((pitch_screw/a_nut)*log((pitch_screw+0.5*a_nut)/(pitch_screw-0.5*a_nut))+0.5*log(4*pitch_screw^2/a_nut^2-1));
flexible_nut_5=((diameter_outer_n^2+D_pitch_n^2)/(diameter_outer_n^2-D_pitch_n^2)+possion_n)*0.5*tan(0.5*angle_screw)^2*(D_pitch_n/pitch_screw)/E_n;
flexible_nut=flexible_nut_1+flexible_nut_2+flexible_nut_3+flexible_nut_4+flexible_nut_5;
%%%
%%%计算丝杠啮合柔度(mm/N)，flexible_screw
flexible_screw_1=(1-possion_s^2)*(3/(4*E_s))*((1-(2-b_screw/a_screw)^2+2*log(a_screw/b_screw))*cot(0.5*angle_screw)^3-4*(c_screw/a_screw)^2*tan(0.5*angle_screw));
flexible_screw_2=(1+possion_s)*(6/(5*E_s))*cot(0.5*angle_screw)*log(a_screw/b_screw);
flexible_screw_3=(1-possion_s^2)*(12*c_screw/(pi*E_s*a_screw^2))*(c_screw-0.5*b_screw*tan(0.5*angle_screw));
flexible_screw_4=(1-possion_s^2)*(2/(pi*E_s))*((pitch_screw/a_screw)*log((pitch_screw+0.5*a_screw)/(pitch_screw-0.5*a_screw))+0.5*log(4*pitch_screw^2/a_screw^2-1));
flexible_screw_5=(1-possion_s)*0.5*tan(0.5*angle_screw)^2*(D_pitch_s/pitch_screw)/E_s;
flexible_screw=flexible_screw_1+flexible_screw_2+flexible_screw_3+flexible_screw_4+flexible_screw_5;
%%%
%%%计算滚柱啮合柔度(mm/N)，flexible_roller
flexible_roller_1=(1-possion_r^2)*(3/(4*E_r))*((1-(2-b_roller/a_roller)^2+2*log(a_roller/b_roller))*cot(0.5*angle_screw)^3-4*(c_roller/a_roller)^2*tan(0.5*angle_screw));
flexible_roller_2=(1+possion_r)*(6/(5*E_r))*cot(0.5*angle_screw)*log(a_roller/b_roller);
flexible_roller_3=(1-possion_r^2)*(12*c_roller/(pi*E_r*a_roller^2))*(c_roller-0.5*b_roller*tan(0.5*angle_screw));
flexible_roller_4=(1-possion_r^2)*(2/(pi*E_r))*((pitch_screw/a_roller)*log((pitch_screw+0.5*a_roller)/(pitch_screw-0.5*a_roller))+0.5*log(4*pitch_screw^2/a_roller^2-1));
flexible_roller_5=(1-possion_r)*0.5*tan(0.5*angle_screw)^2*(D_pitch_r/pitch_screw)/E_r;
flexible_roller=flexible_roller_1+flexible_roller_2+flexible_roller_3+flexible_roller_4+flexible_roller_5;
%%%
%%%计算内螺纹的啮合柔度(mm/N)，flexible_inner
flexible_inner=abs(flexible_nut+flexible_roller)/1;
%%%
%%%计算外螺纹的啮合柔度(mm/N)，flexible_outer
flexible_outer=abs(flexible_screw+flexible_roller)/1;
end
%%%程序结束

%%%计算行星滚柱丝杠的势能
function [potential_total,potential_axial,potential_bend,potential_compress]=potential_roller_screw(order_roller_screw,roller_screw_initial,type_load,Fn_nut,Fn_screw,ratio_nut,ratio_screw,flexible_inner,flexible_outer,B_inner,B_outer,half_width_a_inner,half_width_b_inner,stress_inner,half_width_a_outer,half_width_b_outer,stress_outer)
%%%参数输出
%行星滚柱丝杠总的势能，potential_total
%螺母/丝杠/滚柱轴向拉伸/压缩势能总和，potential_axial
%螺纹弯曲势能总和，potential_bend
%螺纹压缩势能总和，potential_compress
%%%参数输入
%行星滚柱丝杠的序号，order_roller_screw
%行星滚柱丝杠基本参数，roller_screw_initial，[1 行星滚柱丝杠序号 2 行星滚柱丝杠的类型(1表示标准式，2表示反向式),type_roller_screw 3 滚柱个数,number_roller 4 丝杠螺纹头数,N_thread_s 5 螺母螺纹头数,N_thread_n 6 滚柱螺纹头数,N_thread_r
%7 丝杠螺纹旋向(-1表示左旋，1表示右旋),direct_screw 8 螺母螺纹旋向(-1表示左旋，1表示右旋),direct_nut 9 滚柱螺纹旋向(-1表示左旋，1表示右旋),direct_roller 10 丝杠螺纹升角(rad)，rise_angle_s 11 螺母螺纹升角(rad)，rise_angle_n 12 滚柱螺纹升角(rad)，rise_angle_r
%13 第1个滚柱与X轴夹角(rad)，angle_roller 14 内螺纹当量摩擦系数，f_inner 15 外螺纹当量摩擦系数，f_outer 16 螺距(mm)，pitch_screw 17 牙型角(rad)，angle_screw 18 滚柱牙型圆弧半径(mm)，radius_roller 19 滚柱齿廓圆心截面坐标系X轴值(mm)，X_center_r 20 滚柱齿廓圆心截面坐标系Y轴值(mm)，Y_center_r
%21 丝杠螺纹削顶系数，cutter_top_s 22 丝杠螺纹削根系数，cutter_bottom_s 23 螺母螺纹削顶系数，cutter_top_n 24 螺母螺纹削根系数，cutter_bottom_n 25 滚柱螺纹削顶系数，cutter_top_r 26 滚柱螺纹削根系数，cutter_bottom_r 27 丝杠螺纹半牙减薄量(mm)，reduce_s 28 螺母螺纹半牙减薄量(mm)，reduce_n 29 滚柱螺纹半牙减薄量(mm)，reduce_r 30 空
%31 丝杠的螺纹齿顶高(mm)，addendum_s 32 丝杠的螺纹齿根高(mm)，dedendum_s 33 螺母的螺纹齿顶高(mm)，addendum_n 34 螺母的螺纹齿根高(mm)，dedendum_n 35 滚柱的螺纹齿顶高(mm)，addendum_r 36 滚柱的螺纹齿根高(mm)，dedendum_r
%37 丝杠实际顶径(mm)，D_top_s 38 丝杠实际根径(mm)，D_bottom_s 39 螺母实际顶径(mm)，D_top_n
%40 螺母实际根径(mm)，D_bottom_n 41 滚柱实际顶径(mm)，D_top_r 42 滚柱实际根径(mm)，D_bottom_r
%43 丝杠中径(mm),D_pitch_s ,44 螺母中径(mm),D_pitch_n, 45 滚柱中径(mm)，D_pitch_r
%46 丝杠轴长度(mm)，length_shaft_s 47 螺母轴长度(mm)，length_shaft_n 48 滚柱轴长度(mm)，length_shaft_r 49 丝杠轴外径(mm)，diameter_outer_s 50 螺母轴外径(mm)，diameter_outer_n 51 滚柱轴外径(mm)，diameter_outer_r 52 丝杠轴内径(mm)，diameter_inner_s 53 螺母轴内径(mm)，diameter_inner_n 54 滚柱轴内径(mm)，diameter_inner_r
%55 丝杠螺纹长度(mm)，length_thread_s 56 螺母螺纹长度(mm)，length_thread_n 57 滚柱螺纹长度(mm)，length_thread_r 58 内啮合的有效螺纹长度(mm)，length_mesh_inner 59 外啮合的有效螺纹长度(mm)，length_mesh_outer 60 空
%61 左齿轮宽度(mm)，width_gear_left 62 右齿轮宽度(mm)，width_gear_left 63 左保持架宽度(mm)，width_carrier_left 64 左保持架宽度(mm)，width_carrier_right
%65 螺母轴左端相对于丝杠轴左端的位置(mm)，delta_n_s 66 滚柱轴左端相对于丝杠轴左端的位置(mm)，delta_r_s 67 丝杠螺纹左端相对于丝杠轴左端的位置(mm),delta_thread_s 68 螺母螺纹左端相对于螺母轴左端的位置(mm),delta_thread_n 69 滚柱螺纹左端相对于滚柱轴左端的位置(mm),delta_thread_r 70 轴向预紧载荷(N)，preload_axial
%71 丝杠轴的密度(t/mm^-9)，density_s 72 螺母轴的密度(t/mm^-9)，density_n 73 滚柱轴的密度(t/mm^-9)，density_r 74 丝杠弹性模量(Mpa)，E_s 75 螺母弹性模量(Mpa)，E_n 76 滚柱弹性模量(Mpa)，E_r 77 丝杠泊松比，possion_s 78 螺母泊松比，possion_n 79 滚柱泊松比，possion_r
%80 左齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_left 81 右齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_right 82 左行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_left 83 右行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_right
%84 左保持架外径(mm)，outer_carrier_left 85 左保持架内径(mm)，inner_carrier_left 86 右保持架外径(mm)，outer_carrier_right 87 右保持架内径(mm)，inner_carrier_right 88 行星滚柱丝是否指定效率(0表示不指定，1表示指定),sign_efficiency 89 正向驱动效率(无预紧载荷)，advance_efficiency 90 逆向驱动效率(无预紧载荷)，reverse_efficiency
%91 齿轮法向模数(mm)，m_n 92 内齿圈齿数，z_r 93 太阳轮齿数，z_s 94 行星轮齿数，z_p 95 压力角(rad)，press_angle 96 端面压力角(rad)，tran_press_angle 97 螺旋角(rad)，helix_angle 98 内齿圈法向变位系数，n_r 99 太阳轮法向变位系数，n_s 100 行星轮法向变位系数，n_p
%101 工作中心距(mm)，work_center 102 行星轮系内啮合角，mesh_angle(rad) 103 行星轮系外啮合角，mesh_angle(rad) 104 内齿圈齿顶圆，tran_ad_dia_r（mm） 105 太阳轮齿顶圆，tran_ad_dia_s（mm） 106 行星轮齿顶圆，tran_ad_dia_p（mm） 107 内齿圈齿根圆，tran_de_dia_r（mm） 108 太阳轮齿根圆，tran_de_dia_s（mm） 109 行星轮齿根圆，tran_de_dia_p（mm）]
%110 内齿圈的节圆（mm），tran_pitch_dia_r 111 太阳轮的节圆（mm），tran_pitch_dia_s 112 行星轮的节圆（mm），tran_pitch_dia_p 113-120 空
%121 螺母幅板宽度(mm)，width_web_n 122 螺母幅板中心距螺母轴左端距离(mm)，delta_web_n 123 丝杠左幅板宽度(mm)，width_left_web_s 124 丝杠左幅板中心距丝杠轴左端距离(mm)，delta_left_web_s 125 丝杠右幅板宽度(mm)，width_right_web_s 126 丝杠右幅板中心距丝杠轴左端距离(mm)，delta_right_web_s 127-130 空
%131 螺母轮缘轴序号,shaft_self_nut 132 丝杠轮缘轴序号，shaft_self_screw 133 滚柱轮缘轴序号，shaft_self_roller,134-140 空
%141 行星轮刀具齿顶高系数，142 行星轮刀具顶隙系数，143 行星轮齿顶削减量，144 内齿圈刀具齿顶高系数，145 内齿圈刀具顶隙系数，146 内齿圈齿顶削减量,147 太阳轮刀具齿顶高系数，148 太阳轮刀具顶隙系数，149 太阳轮齿顶削减量
%150-160 空 161 有效内啮合螺纹相对于螺母轴左端的位置(mm)，thread_inner_nut 162 有效内啮合螺纹相对于滚柱轴左端的位置(mm)，thread_inner_roller 163 有效外啮合螺纹相对于丝杠轴左端的位置(mm)，thread_outer_screw 164 有效外啮合螺纹相对于滚柱轴左端的位置(mm)，thread_outer_roller]
%行星滚柱丝杠受载方式，type_load，1表示同侧受载，2表示异侧受载
%螺母的接触点法向载荷的向量，Fn_nut，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
%丝杠的接触点法向载荷的向量，Fn_screw，[1 接触点序号(注:序号从螺母受载端开始) 2至1+滚柱个数 接触点法向载荷(N)]
%螺母轴向分量比例，ratio_nut
%丝杠轴向分量比例，ratio_screw
%内螺纹的啮合柔度(mm/N)，flexible_inner
%外螺纹的啮合柔度(mm/N)，flexible_outer
%内啮合压缩势能辅助值，B_inner
%外啮合压缩势能辅助值，B_outer
%内啮合的接触长半轴(mm)，half_width_a_inner
%内啮合的接触短半轴(mm)，half_width_b_inner
%内啮合接触应力(Mpa)，stress_inner
%外啮合的接触长半轴(mm)，half_width_a_outer
%外啮合的接触短半轴(mm)，half_width_b_outer
%外啮合接触应力(Mpa)，stress_outer
%%%程序开始
%%%生成行星滚柱丝杠轮缘轴的网格段矩阵
[roller_screw_segment,number]=roller_screw_segment_produce(order_roller_screw,roller_screw_initial);
%行星滚柱丝杠轮缘轴的网格段矩阵，roller_screw_segment，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
%有效螺纹啮合牙纹数，number
%%%计算行星滚柱丝杠的弯曲、拉压、扭转的全局坐标系的刚度矩阵，shaft_stiffness
[roller_screw_stiffness]=roller_screw_stiffness_produce(order_roller_screw,roller_screw_segment);
%行星滚柱丝杠的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，roller_screw_stiffness，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
%%%记录行星滚柱丝杠相关参数
%行星滚柱丝杠序号，order_roller_screw
id_roller_screw=(roller_screw_initial(:,1)==order_roller_screw);
roller_screw_temp=roller_screw_initial(id_roller_screw,:);
%%%记录滚柱个数，number_roller
number_roller=roller_screw_temp(1,3);
%%%
%%%计算螺母拉伸/压缩势能，potential_nut
%%%计算螺母的轴向载荷向量，Axial_nut
Axial_nut=zeros(6*(2+number),1);
for ii=1:number %有效螺纹啮合牙纹数，number
    Axial_nut(6*ii+3,1)=-1*sum(Fn_nut(ii,2:1+number_roller))*ratio_nut;
end
Axial_nut(3,1)=sum(sum(Fn_nut(:,2:1+number_roller)))*ratio_nut;
%%%记录螺母的刚度矩阵，stiffness_nut
id_stiffness_nut=(roundn(roller_screw_stiffness(:,1),-9)==roundn(fix(order_roller_screw)+0.0001,-9));
stiffness_nut=roller_screw_stiffness(id_stiffness_nut,2:1+6*(2+number));
%%%计算螺母的柔度矩阵，flexible_nut
flexible_nut=stiffness_nut^-1;
%%%计算螺母拉伸/压缩势能，potential_nut
potential_nut=0.5*Axial_nut'*flexible_nut*Axial_nut;
%%%
%%%计算丝杠拉伸/压缩势能，potential_screw
%%%计算丝杠的轴向载荷向量，Axial_screw
Axial_screw=zeros(6*(2+number),1);
for ii=1:number %有效螺纹啮合牙纹数，number
    Axial_screw(6*ii+3,1)=sum(Fn_screw(ii,2:1+number_roller))*ratio_screw;
end
if type_load==1 %行星滚柱丝杠受载方式，type_load，1表示同侧受载，2表示异侧受载
    Axial_screw(3,1)=-1*sum(sum(Fn_screw(:,2:1+number_roller)))*ratio_screw;
elseif type_load==2 %行星滚柱丝杠受载方式，type_load，1表示同侧受载，2表示异侧受载
    Axial_screw(6*(1+number)+3,1)=-1*sum(sum(Fn_screw(:,2:1+number_roller)))*ratio_screw;
end
%%%记录丝杠的刚度矩阵，stiffness_screw
id_stiffness_screw=(roundn(roller_screw_stiffness(:,1),-9)==roundn(fix(order_roller_screw)+0.0002,-9));
stiffness_screw=roller_screw_stiffness(id_stiffness_screw,2:1+6*(2+number));
%%%计算丝杠的柔度矩阵，flexible_screw
flexible_screw=stiffness_screw^-1;
%%%计算丝杠拉伸/压缩势能，potential_screw
potential_screw=0.5*Axial_screw'*flexible_screw*Axial_screw;
%%%
%%%定义滚柱拉伸/压缩势能，potential_roller
potential_roller=0;
%%%记录滚柱的刚度矩阵，stiffness_roller
id_stiffness_roller=(roundn(roller_screw_stiffness(:,1),-9)==roundn(fix(order_roller_screw)+0.0004,-9));
stiffness_roller=roller_screw_stiffness(id_stiffness_roller,2:1+6*(2+number));
%%%计算滚柱的柔度矩阵，flexible_roller
flexible_roller=stiffness_roller^-1;
%%%计算滚柱拉伸/压缩势能，potential_screw
for kk=1:number_roller %滚柱个数，number_roller
    Axial_roller=zeros(6*(2+number),1);
    for ii=1:number %有效螺纹啮合牙纹数，number
        Axial_roller(6*ii+3,1)=Fn_nut(ii,1+kk)*ratio_nut-Fn_screw(ii,1+kk)*ratio_screw;
    end
    %%%计算滚柱拉伸/压缩势能，potential_roller
    potential_roller=potential_roller+0.5*Axial_roller'*flexible_roller*Axial_roller;
end
%%%
%%%计算螺母/丝杠/滚柱轴向拉伸/压缩势能总和，potential_axial
potential_axial=potential_nut+potential_screw+potential_roller;
%%%
%%%计算内啮合螺纹弯曲势能，potential_bend_inner
%内螺纹的啮合柔度(mm/N)，flexible_inner
potential_bend_inner=sum(sum(0.5*Fn_nut(1:number,2:1+number_roller).*flexible_inner.*Fn_nut(1:number,2:1+number_roller)));
%%%计算外啮合螺纹弯曲势能，potential_bend_outer
%外螺纹的啮合柔度(mm/N)，flexible_outer
potential_bend_outer=sum(sum(0.5*Fn_screw(1:number,2:1+number_roller).*flexible_outer.*Fn_screw(1:number,2:1+number_roller)));
%%%计算螺纹弯曲势能总和，potential_bend
potential_bend=potential_bend_inner+potential_bend_outer;
%%%
%%%计算内啮合螺纹压缩势能，potential_compress_inner
%内啮合压缩势能辅助值，B_inner
%内啮合的接触长半轴(mm)，half_width_a_inner
%内啮合的接触短半轴(mm)，half_width_b_inner
%内啮合接触应力(Mpa)，stress_inner
potential_compress_inner=sum(sum(0.25*pi*B_inner*half_width_a_inner(1:number,2:1+number_roller).*(half_width_b_inner(1:number,2:1+number_roller).^2).*stress_inner(1:number,2:1+number_roller)));
%%%计算外啮合螺纹压缩势能，potential_compress_outer
%外啮合压缩势能辅助值，B_outer
%外啮合的接触长半轴(mm)，half_width_a_outer
%外啮合的接触短半轴(mm)，half_width_b_outer
%外啮合接触应力(Mpa)，stress_outer
potential_compress_outer=sum(sum(0.25*pi*B_outer*half_width_a_outer(1:number,2:1+number_roller).*(half_width_b_outer(1:number,2:1+number_roller).^2).*stress_outer(1:number,2:1+number_roller)));
%%%计算螺纹压缩势能总和，potential_compress
potential_compress=potential_compress_inner+potential_compress_outer;
%%%
%%%计算行星滚柱丝杠总的势能，potential_total
potential_total=potential_axial+potential_bend+potential_compress;
end
%%%程序结束

%%%生成行星滚柱丝杠轮缘轴的网格段矩阵
function [roller_screw_segment,number]=roller_screw_segment_produce(order_roller_screw,roller_screw_initial)
%%%参数输出
%行星滚柱丝杠轮缘轴的网格段矩阵，roller_screw_segment，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
%有效螺纹啮合牙纹数，number
%%%参数输入
%行星滚柱丝杠序号，order_roller_screw
%行星滚柱丝杠基本参数，roller_screw_initial，[1 行星滚柱丝杠序号 2 行星滚柱丝杠的类型(1表示标准式，2表示反向式),type_roller_screw 3 滚柱个数,number_roller 4 丝杠螺纹头数,N_thread_s 5 螺母螺纹头数,N_thread_n 6 滚柱螺纹头数,N_thread_r
%7 丝杠螺纹旋向(-1表示左旋，1表示右旋),direct_screw 8 螺母螺纹旋向(-1表示左旋，1表示右旋),direct_nut 9 滚柱螺纹旋向(-1表示左旋，1表示右旋),direct_roller 10 丝杠螺纹升角(rad)，rise_angle_s 11 螺母螺纹升角(rad)，rise_angle_n 12 滚柱螺纹升角(rad)，rise_angle_r
%13 第1个滚柱与X轴夹角(rad)，angle_roller 14 内螺纹当量摩擦系数，f_inner 15 外螺纹当量摩擦系数，f_outer 16 螺距(mm)，pitch_screw 17 牙型角(rad)，angle_screw 18 滚柱牙型圆弧半径(mm)，radius_roller 19 滚柱齿廓圆心截面坐标系X轴值(mm)，X_center_r 20 滚柱齿廓圆心截面坐标系Y轴值(mm)，Y_center_r
%21 丝杠螺纹削顶系数，cutter_top_s 22 丝杠螺纹削根系数，cutter_bottom_s 23 螺母螺纹削顶系数，cutter_top_n 24 螺母螺纹削根系数，cutter_bottom_n 25 滚柱螺纹削顶系数，cutter_top_r 26 滚柱螺纹削根系数，cutter_bottom_r 27 丝杠螺纹半牙减薄量(mm)，reduce_s 28 螺母螺纹半牙减薄量(mm)，reduce_n 29 滚柱螺纹半牙减薄量(mm)，reduce_r 30 空
%31 丝杠的螺纹齿顶高(mm)，addendum_s 32 丝杠的螺纹齿根高(mm)，dedendum_s 33 螺母的螺纹齿顶高(mm)，addendum_n 34 螺母的螺纹齿根高(mm)，dedendum_n 35 滚柱的螺纹齿顶高(mm)，addendum_r 36 滚柱的螺纹齿根高(mm)，dedendum_r
%37 丝杠实际顶径(mm)，D_top_s 38 丝杠实际根径(mm)，D_bottom_s 39 螺母实际顶径(mm)，D_top_n
%40 螺母实际根径(mm)，D_bottom_n 41 滚柱实际顶径(mm)，D_top_r 42 滚柱实际根径(mm)，D_bottom_r
%43 丝杠中径(mm),D_pitch_s ,44 螺母中径(mm),D_pitch_n, 45 滚柱中径(mm)，D_pitch_r
%46 丝杠轴长度(mm)，length_shaft_s 47 螺母轴长度(mm)，length_shaft_n 48 滚柱轴长度(mm)，length_shaft_r 49 丝杠轴外径(mm)，diameter_outer_s 50 螺母轴外径(mm)，diameter_outer_n 51 滚柱轴外径(mm)，diameter_outer_r 52 丝杠轴内径(mm)，diameter_inner_s 53 螺母轴内径(mm)，diameter_inner_n 54 滚柱轴内径(mm)，diameter_inner_r
%55 丝杠螺纹长度(mm)，length_thread_s 56 螺母螺纹长度(mm)，length_thread_n 57 滚柱螺纹长度(mm)，length_thread_r 58 内啮合的有效螺纹长度(mm)，length_mesh_inner 59 外啮合的有效螺纹长度(mm)，length_mesh_outer 60 空
%61 左齿轮宽度(mm)，width_gear_left 62 右齿轮宽度(mm)，width_gear_left 63 左保持架宽度(mm)，width_carrier_left 64 左保持架宽度(mm)，width_carrier_right
%65 螺母轴左端相对于丝杠轴左端的位置(mm)，delta_n_s 66 滚柱轴左端相对于丝杠轴左端的位置(mm)，delta_r_s 67 丝杠螺纹左端相对于丝杠轴左端的位置(mm),delta_thread_s 68 螺母螺纹左端相对于螺母轴左端的位置(mm),delta_thread_n 69 滚柱螺纹左端相对于滚柱轴左端的位置(mm),delta_thread_r 70 轴向预紧载荷(N)，preload_axial
%71 丝杠轴的密度(t/mm^-9)，density_s 72 螺母轴的密度(t/mm^-9)，density_n 73 滚柱轴的密度(t/mm^-9)，density_r 74 丝杠弹性模量(Mpa)，E_s 75 螺母弹性模量(Mpa)，E_n 76 滚柱弹性模量(Mpa)，E_r 77 丝杠泊松比，possion_s 78 螺母泊松比，possion_n 79 滚柱泊松比，possion_r
%80 左齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_left 81 右齿轮左端相对于滚柱轴左端位置(mm)，delta_gear_right 82 左行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_left 83 右行星架左端相对于滚柱轴左端位置(mm)，delta_carrier_right
%84 左保持架外径(mm)，outer_carrier_left 85 左保持架内径(mm)，inner_carrier_left 86 右保持架外径(mm)，outer_carrier_right 87 右保持架内径(mm)，inner_carrier_right 88 行星滚柱丝是否指定效率(0表示不指定，1表示指定),sign_efficiency 89 正向驱动效率(无预紧载荷)，advance_efficiency 90 逆向驱动效率(无预紧载荷)，reverse_efficiency
%91 齿轮法向模数(mm)，m_n 92 内齿圈齿数，z_r 93 太阳轮齿数，z_s 94 行星轮齿数，z_p 95 压力角(rad)，press_angle 96 端面压力角(rad)，tran_press_angle 97 螺旋角(rad)，helix_angle 98 内齿圈法向变位系数，n_r 99 太阳轮法向变位系数，n_s 100 行星轮法向变位系数，n_p
%101 工作中心距(mm)，work_center 102 行星轮系内啮合角，mesh_angle(rad) 103 行星轮系外啮合角，mesh_angle(rad) 104 内齿圈齿顶圆，tran_ad_dia_r（mm） 105 太阳轮齿顶圆，tran_ad_dia_s（mm） 106 行星轮齿顶圆，tran_ad_dia_p（mm） 107 内齿圈齿根圆，tran_de_dia_r（mm） 108 太阳轮齿根圆，tran_de_dia_s（mm） 109 行星轮齿根圆，tran_de_dia_p（mm）]
%110 内齿圈的节圆（mm），tran_pitch_dia_r 111 太阳轮的节圆（mm），tran_pitch_dia_s 112 行星轮的节圆（mm），tran_pitch_dia_p 113-120 空
%121 螺母幅板宽度(mm)，width_web_n 122 螺母幅板中心距螺母轴左端距离(mm)，delta_web_n 123 丝杠左幅板宽度(mm)，width_left_web_s 124 丝杠左幅板中心距丝杠轴左端距离(mm)，delta_left_web_s 125 丝杠右幅板宽度(mm)，width_right_web_s 126 丝杠右幅板中心距丝杠轴左端距离(mm)，delta_right_web_s 127-130 空
%131 螺母轮缘轴序号,shaft_self_nut 132 丝杠轮缘轴序号，shaft_self_screw 133 滚柱轮缘轴序号，shaft_self_roller,134-140 空
%141 行星轮刀具齿顶高系数，142 行星轮刀具顶隙系数，143 行星轮齿顶削减量，144 内齿圈刀具齿顶高系数，145 内齿圈刀具顶隙系数，146 内齿圈齿顶削减量,147 太阳轮刀具齿顶高系数，148 太阳轮刀具顶隙系数，149 太阳轮齿顶削减量
%150-160 空 161 有效内啮合螺纹相对于螺母轴左端的位置(mm)，thread_inner_nut 162 有效内啮合螺纹相对于滚柱轴左端的位置(mm)，thread_inner_roller 163 有效外啮合螺纹相对于丝杠轴左端的位置(mm)，thread_outer_screw 164 有效外啮合螺纹相对于滚柱轴左端的位置(mm)，thread_outer_roller]
%%%程序开始
%%%定义行星滚柱丝杠轮缘轴的网格段矩阵，roller_screw_segment，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
roller_screw_segment=zeros(0,16);
%%%记录序号为order_roller_screw的行星滚柱丝杠的基本参数
%行星滚柱丝杠序号，order_roller_screw
id_roller_screw=(roller_screw_initial(:,1)==order_roller_screw);
roller_screw_temp=roller_screw_initial(id_roller_screw,:);
%%%记录行星滚柱丝杠的螺矩(mm)，pitch_screw
pitch_screw=roller_screw_temp(1,16);
%%%记录丝杠实际根径(mm)，D_bottom_s
D_bottom_s=roller_screw_temp(1,38);
%%%记录螺母实际根径(mm)，D_bottom_n
D_bottom_n=roller_screw_temp(1,40);
%%%记录滚柱实际根径(mm)，D_bottom_r
D_bottom_r=roller_screw_temp(1,42);
%%%记录丝杠轴长度(mm)，length_shaft_s
length_shaft_s=roller_screw_temp(1,46);
%%%记录螺母轴长度(mm)，length_shaft_n
length_shaft_n=roller_screw_temp(1,47);
%%%记录滚柱轴长度(mm)，length_shaft_r
length_shaft_r=roller_screw_temp(1,48);
%%%记录螺母轴外径(mm)，diameter_outer_n
diameter_outer_n=roller_screw_temp(1,50);
%%%记录丝杠轴内径(mm)，diameter_inner_s
diameter_inner_s=roller_screw_temp(1,52);
%%%记录滚柱轴内径(mm)，diameter_inner_r
diameter_inner_r=roller_screw_temp(1,54);
%%%记录内啮合的有效螺纹长度(mm)，length_mesh_inner
length_mesh_inner=roller_screw_temp(1,58);
%%%记录外啮合的有效螺纹长度(mm)，length_mesh_outer
length_mesh_outer=roller_screw_temp(1,59);
%%%记录螺母轴左端相对于丝杠轴左端的位置(mm)，delta_n_s
delta_n_s=roller_screw_temp(1,65);
%%%记录滚柱轴左端相对于丝杠轴左端的位置(mm)，delta_r_s
delta_r_s=roller_screw_temp(1,66);
%%%记录丝杠螺纹左端相对于丝杠轴左端的位置(mm),delta_thread_s
delta_thread_s=roller_screw_temp(1,67);
%%%记录螺母螺纹左端相对于螺母轴左端的位置(mm),delta_thread_n
delta_thread_n=roller_screw_temp(1,68);
%%%记录滚柱螺纹左端相对于滚柱轴左端的位置(mm),delta_thread_r
delta_thread_r=roller_screw_temp(1,69);
%%%记录丝杠轴的密度(t/mm^-9)，density_s
density_s=roller_screw_temp(1,71);
%%%记录螺母轴的密度(t/mm^-9)，density_n
density_n=roller_screw_temp(1,72);
%%%记录滚柱轴的密度(t/mm^-9)，density_r
density_r=roller_screw_temp(1,73);
%%%记录丝杠弹性模量(Mpa)，E_s
E_s=roller_screw_temp(1,74);
%%%记录螺母弹性模量(Mpa)，E_n
E_n=roller_screw_temp(1,75);
%%%记录滚柱弹性模量(Mpa)，E_r
E_r=roller_screw_temp(1,76);
%%%记录丝杠泊松比，possion_s
possion_s=roller_screw_temp(1,77);
%%%记录螺母泊松比，possion_n
possion_n=roller_screw_temp(1,78);
%%%记录滚柱泊松比，possion_r
possion_r=roller_screw_temp(1,79);
%%%记录螺母轮缘轴序号,shaft_self_nut
shaft_self_nut=roller_screw_temp(1,131);
%%%记录丝杠轮缘轴序号,shaft_self_screw
shaft_self_screw=roller_screw_temp(1,132);
%%%记录滚柱轮缘轴序号,shaft_self_roller
shaft_self_roller=roller_screw_temp(1,133);
%%%
%%%计算有效螺纹长度(mm)，length_mesh
length_mesh=min(length_mesh_inner,length_mesh_outer);
%%%计算有效螺纹啮合齿数，number
number=1+fix(length_mesh/pitch_screw);
%%%计算丝杠螺纹左端相对于丝杠轴左端的位置(mm)，thread_left_s
thread_left_s=delta_thread_s;
%%%计算滚柱螺纹左端相对于丝杠轴左端的位置(mm)，thread_left_r
thread_left_r=delta_thread_r+delta_r_s;
%%%计算螺母螺纹左端相对于丝杠轴左端的位置(mm)，thread_left_n
thread_left_n=delta_thread_n+delta_n_s;
%%%计算螺母有效螺纹左端距螺母轴左端的距离(mm)，distance_nut_left
distance_nut_left=max([thread_left_n,thread_left_s,thread_left_r])-delta_n_s;
%%%计算螺母有效螺纹右端距螺母轴左端的距离(mm)，distance_nut_right
distance_nut_right=distance_nut_left+(number-1)*pitch_screw;
%%%计算丝杠有效螺纹左端距丝杠轴左端的距离(mm)，distance_screw_left
distance_screw_left=max([thread_left_n,thread_left_s,thread_left_r]);
%%%计算丝杠有效螺纹右端距丝杠轴左端的距离(mm)，distance_screw_right
distance_screw_right=distance_screw_left+(number-1)*pitch_screw;
%%%计算滚柱有效螺纹左端距滚柱轴左端的距离(mm)，distance_roller_left
distance_roller_left=max([thread_left_n,thread_left_s,thread_left_r])-delta_r_s;
%%%计算滚柱有效螺纹右端距滚柱轴左端的距离(mm)，distance_roller_right
distance_roller_right=distance_roller_left+(number-1)*pitch_screw;
%%%
%%%生成螺母轮缘轴的网格段矩阵，shaft_segment_nut，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
shaft_segment_nut=zeros(0,16);
for kk=1:number+1 %有效螺纹啮合齿数，number
    if kk==1 %螺母轴非啮合区前段
        shaft_segment_nut(kk,1)=shaft_self_nut;  %螺母轮缘轴序号,shaft_self_nut
        shaft_segment_nut(kk,2)=kk;  %网格段序列号
        shaft_segment_nut(kk,3)=distance_nut_left; %螺母有效螺纹左端距螺母轴左端的距离(mm)，distance_nut_left
        shaft_segment_nut(kk,4)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,5)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
        shaft_segment_nut(kk,6)=E_n;  %螺母弹性模量(Mpa)，E_n
        shaft_segment_nut(kk,7)=possion_n;  %螺母泊松比，possion_n
        shaft_segment_nut(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_nut(kk,9)=distance_nut_left;  %螺母有效螺纹左端距螺母轴左端的距离(mm)，distance_nut_left
        shaft_segment_nut(kk,10)=density_n;  %螺母轴的密度(t/mm^-9)，density_n
        shaft_segment_nut(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_nut(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_nut(kk,13)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,14)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,15)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
        shaft_segment_nut(kk,16)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
    elseif kk==number+1 %螺母轴非啮合区前段
        shaft_segment_nut(kk,1)=shaft_self_nut;  %螺母轮缘轴序号,shaft_self_nut
        shaft_segment_nut(kk,2)=kk;  %网格段序列号
        shaft_segment_nut(kk,3)=length_shaft_n-distance_nut_right; %螺母有效螺纹右端距螺母轴左端的距离(mm)，distance_nut_right
        shaft_segment_nut(kk,4)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,5)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
        shaft_segment_nut(kk,6)=E_n;  %螺母弹性模量(Mpa)，E_n
        shaft_segment_nut(kk,7)=possion_n;  %螺母泊松比，possion_n
        shaft_segment_nut(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_nut(kk,9)=length_shaft_n;  %螺母轴长度(mm)，length_shaft_n
        shaft_segment_nut(kk,10)=density_n;  %螺母轴的密度(t/mm^-9)，density_n
        shaft_segment_nut(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_nut(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_nut(kk,13)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,14)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,15)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
        shaft_segment_nut(kk,16)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
    else
        shaft_segment_nut(kk,1)=shaft_self_nut;  %螺母轮缘轴序号,shaft_self_nut
        shaft_segment_nut(kk,2)=kk;  %网格段序列号
        shaft_segment_nut(kk,3)=pitch_screw; %行星滚柱丝杠的螺矩(mm)，pitch_screw
        shaft_segment_nut(kk,4)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,5)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
        shaft_segment_nut(kk,6)=E_n;  %螺母弹性模量(Mpa)，E_n
        shaft_segment_nut(kk,7)=possion_n;  %螺母泊松比，possion_n
        shaft_segment_nut(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_nut(kk,9)=distance_nut_left+(kk-1)*pitch_screw;  %螺母轴长度(mm)，length_shaft_n
        shaft_segment_nut(kk,10)=density_n;  %螺母轴的密度(t/mm^-9)，density_n
        shaft_segment_nut(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_nut(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_nut(kk,13)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,14)=diameter_outer_n;  %螺母轴外径(mm)，diameter_outer_n
        shaft_segment_nut(kk,15)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
        shaft_segment_nut(kk,16)=D_bottom_n;  %螺母实际根径(mm)，D_bottom_n
    end
end
%%%记录行星滚柱丝杠轮缘轴的网格段矩阵，roller_screw_segment，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
roller_screw_segment(1:kk,:)=shaft_segment_nut;
%%%定义计数器，dd
dd=kk;
%%%
%%%生成丝杠轮缘轴的网格段矩阵，shaft_segment_screw，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
shaft_segment_screw=zeros(0,16);
for kk=1:number+1 %有效螺纹啮合齿数，number
    if kk==1 %丝杠轴非啮合区前段
        shaft_segment_screw(kk,1)=shaft_self_screw;  %丝杠轮缘轴序号,shaft_self_screw
        shaft_segment_screw(kk,2)=kk;  %网格段序列号
        shaft_segment_screw(kk,3)=distance_screw_left; %丝杠有效螺纹左端距丝杠轴左端的距离(mm)，distance_screw_left
        shaft_segment_screw(kk,4)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,5)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
        shaft_segment_screw(kk,6)=E_s;  %丝杠弹性模量(Mpa)，E_s
        shaft_segment_screw(kk,7)=possion_s;  %丝杠泊松比，possion_s
        shaft_segment_screw(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_screw(kk,9)=distance_screw_left;  %丝杠有效螺纹左端距丝杠轴左端的距离(mm)，distance_screw_left
        shaft_segment_screw(kk,10)=density_s;  %丝杠轴的密度(t/mm^-9)，density_s
        shaft_segment_screw(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_screw(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_screw(kk,13)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,14)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,15)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
        shaft_segment_screw(kk,16)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
    elseif kk==number+1 %丝杠轴非啮合区前段
        shaft_segment_screw(kk,1)=shaft_self_screw;  %丝杠轮缘轴序号,shaft_self_screw
        shaft_segment_screw(kk,2)=kk;  %网格段序列号
        shaft_segment_screw(kk,3)=length_shaft_s-distance_screw_right; %丝杠有效螺纹右端距丝杠轴左端的距离(mm)，distance_screw_right
        shaft_segment_screw(kk,4)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,5)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
        shaft_segment_screw(kk,6)=E_s;  %丝杠弹性模量(Mpa)，E_s
        shaft_segment_screw(kk,7)=possion_s;  %丝杠泊松比，possion_s
        shaft_segment_screw(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_screw(kk,9)=length_shaft_s;  %丝杠轴长度(mm)，length_shaft_s
        shaft_segment_screw(kk,10)=density_s;  %丝杠轴的密度(t/mm^-9)，density_s
        shaft_segment_screw(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_screw(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_screw(kk,13)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,14)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,15)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
        shaft_segment_screw(kk,16)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
    else
        shaft_segment_screw(kk,1)=shaft_self_screw;  %丝杠轮缘轴序号,shaft_self_screw
        shaft_segment_screw(kk,2)=kk;  %网格段序列号
        shaft_segment_screw(kk,3)=pitch_screw; %行星滚柱丝杠的螺矩(mm)，pitch_screw
        shaft_segment_screw(kk,4)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,5)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
        shaft_segment_screw(kk,6)=E_s;  %丝杠弹性模量(Mpa)，E_s
        shaft_segment_screw(kk,7)=possion_s;  %丝杠泊松比，possion_s
        shaft_segment_screw(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_screw(kk,9)=distance_screw_left+(kk-1)*pitch_screw;  %丝杠轴长度(mm)，length_shaft_s
        shaft_segment_screw(kk,10)=density_s;  %丝杠轴的密度(t/mm^-9)，density_s
        shaft_segment_screw(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_screw(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_screw(kk,13)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,14)=D_bottom_s;  %丝杠实际根径(mm)，D_bottom_s
        shaft_segment_screw(kk,15)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
        shaft_segment_screw(kk,16)=diameter_inner_s;  %丝杠轴内径(mm)，diameter_inner_s
    end
end
%%%记录行星滚柱丝杠轮缘轴的网格段矩阵，roller_screw_segment，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
roller_screw_segment(dd+1:dd+kk,:)=shaft_segment_screw;
%%%更新计数器，dd
dd=dd+kk;
%%%
%%%生成滚柱轮缘轴的网格段矩阵，shaft_segment_roller，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
shaft_segment_roller=zeros(0,16);
for kk=1:number+1 %有效螺纹啮合齿数，number
    if kk==1 %滚柱轴非啮合区前段
        shaft_segment_roller(kk,1)=shaft_self_roller;  %滚柱轮缘轴序号,shaft_self_roller
        shaft_segment_roller(kk,2)=kk;  %网格段序列号
        shaft_segment_roller(kk,3)=distance_roller_left; %滚柱有效螺纹左端距滚柱轴左端的距离(mm)，distance_roller_left
        shaft_segment_roller(kk,4)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,5)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
        shaft_segment_roller(kk,6)=E_r;  %滚柱弹性模量(Mpa)，E_r
        shaft_segment_roller(kk,7)=possion_r;  %滚柱泊松比，possion_r
        shaft_segment_roller(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_roller(kk,9)=distance_roller_left;  %滚柱有效螺纹左端距滚柱轴左端的距离(mm)，distance_roller_left
        shaft_segment_roller(kk,10)=density_r;  %滚柱轴的密度(t/mm^-9)，density_r
        shaft_segment_roller(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_roller(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_roller(kk,13)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,14)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,15)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
        shaft_segment_roller(kk,16)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
    elseif kk==number+1 %滚柱轴非啮合区前段
        shaft_segment_roller(kk,1)=shaft_self_roller;  %滚柱轮缘轴序号,shaft_self_roller
        shaft_segment_roller(kk,2)=kk;  %网格段序列号
        shaft_segment_roller(kk,3)=length_shaft_r-distance_roller_right; %滚柱有效螺纹右端距滚柱轴左端的距离(mm)，distance_roller_right
        shaft_segment_roller(kk,4)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,5)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
        shaft_segment_roller(kk,6)=E_r;  %滚柱弹性模量(Mpa)，E_r
        shaft_segment_roller(kk,7)=possion_r;  %滚柱泊松比，possion_r
        shaft_segment_roller(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_roller(kk,9)=length_shaft_r;  %滚柱轴长度(mm)，length_shaft_r
        shaft_segment_roller(kk,10)=density_r;  %滚柱轴的密度(t/mm^-9)，density_r
        shaft_segment_roller(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_roller(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_roller(kk,13)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,14)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,15)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
        shaft_segment_roller(kk,16)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
    else
        shaft_segment_roller(kk,1)=shaft_self_roller;  %滚柱轮缘轴序号,shaft_self_roller
        shaft_segment_roller(kk,2)=kk;  %网格段序列号
        shaft_segment_roller(kk,3)=pitch_screw; %行星滚柱滚柱的螺矩(mm)，pitch_screw
        shaft_segment_roller(kk,4)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,5)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
        shaft_segment_roller(kk,6)=E_r;  %滚柱弹性模量(Mpa)，E_r
        shaft_segment_roller(kk,7)=possion_r;  %滚柱泊松比，possion_r
        shaft_segment_roller(kk,8)=0.886;  %8 网格段剪切变形因子
        shaft_segment_roller(kk,9)=distance_roller_left+(kk-1)*pitch_screw;  %滚柱轴长度(mm)，length_shaft_r
        shaft_segment_roller(kk,10)=density_r;  %滚柱轴的密度(t/mm^-9)，density_r
        shaft_segment_roller(kk,11)=1;  %11 网格段右节点是否为固有节点（0表示非固有 1表示固有）
        shaft_segment_roller(kk,12)=0;  %12 网格段右节点装配数量
        shaft_segment_roller(kk,13)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,14)=D_bottom_r;  %滚柱实际根径(mm)，D_bottom_r
        shaft_segment_roller(kk,15)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
        shaft_segment_roller(kk,16)=diameter_inner_r;  %滚柱轴内径(mm)，diameter_inner_r
    end
end
%%%记录行星滚柱丝杠轮缘轴的网格段矩阵，roller_screw_segment，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
roller_screw_segment(dd+1:dd+kk,:)=shaft_segment_roller;
%%%更新计数器，dd
dd=dd+kk;
%%%
%%%更新轴的刚度矩阵
for ii=1:dd %计数器，dd
    poisson=roller_screw_segment(ii,7);
    diameter_outer=roller_screw_segment(ii,4);
    diameter_inner=roller_screw_segment(ii,5);
    diameter_ratio=diameter_inner/diameter_outer;
    a_factor=6*(1+poisson)*(1+diameter_ratio^2)^2/((7+6*poisson)*(1+diameter_ratio^2)^2+(20+12*poisson)*diameter_ratio^2);
    roller_screw_segment(ii,8)=a_factor;
end
end
%%%程序结束

%%%计算行星滚柱丝杠的弯曲、拉压、扭转的全局坐标系的刚度矩阵，shaft_stiffness
function [roller_screw_stiffness]=roller_screw_stiffness_produce(order_roller_screw,roller_screw_segment)
%%%参数输出
%行星滚柱丝杠的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，roller_screw_stiffness，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
%%%参数输入
%行星滚柱丝杠的序号，order_roller_screw
%轴系的轴的网格段参数矩阵，roller_screw_segment，[1 轴的序号 2 网格段序列号 3 网格段长度(mm) 4 网格段平均外径(mm) 5 网格段平均内径(mm) 6 网格段弹性模量(Mpa) 7 网格段泊松比 8 网格段剪切变形因子 9 网格段右端距轴左端距离(mm) 10 网格段密度(t/mm3) 11 网格段右节点是否为固有节点（0表示非固有 1表示固有） 12 网格段右节点装配个数 13 网格段左端外径(mm) 14 网络段右端外径(mm) 15 网络段左端内径(mm) 16 网络段右端(mm) ]
%%%程序开始
%%%定义行星滚柱丝杠的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，roller_screw_stiffness，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
roller_screw_stiffness=zeros(0,1);
%%%定义计数器，dd
dd=0;
%%%
for mm=1:3
    %%%计算螺母/丝杠/滚柱的轮缘轴的序号，order_shaft
    if mm~=3
        %%%计算螺母/丝杠的轮缘轴的序号，order_shaft
        order_shaft=fix(order_roller_screw)+0.0001*mm; %行星滚柱丝杠的序号，order_roller_screw
    elseif mm==3
        %%%计算螺滚柱的轮缘轴的序号，order_shaft
        order_shaft=fix(order_roller_screw)+0.0001*(mm+1); %行星滚柱丝杠的序号，order_roller_screw
    end
    %记录序号为order_roller_screw轴的网格段矩阵，shaft_segment_single，【1 网格段序列号 2 网格段长度(mm) 3 网格段平均外径(mm) 4 网格段平均内径(mm) 5 网格段弹性模量(Mpa) 6 网格段泊松比 7 网格段剪切变形因子 8 网格段右端距轴左端距离(mm) 9 网格段密度(t/mm3)】
    id_shaft_segment=(roundn(roller_screw_segment(:,1),-9)==roundn(order_shaft,-9));
    shaft_segment_single=roller_screw_segment(id_shaft_segment,2:10);
    %按网格段序号重新排列单个轴的网格段矩阵
    shaft_segment_single=sortrows(shaft_segment_single,1);
    %计算序号为order_shaft轴的节点数，shaft_segment
    shaft_segment=sum(id_shaft_segment)+1;
    %%%计算序号为order_shaft单个轴的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，shaft_single_stiffness，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    shaft_single_stiffness=zeros(6*shaft_segment,6*shaft_segment);
    %计算序号为order_shaft单个轴的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，shaft_single_stiffness_local，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness_local*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    shaft_single_stiffness_local=zeros(6*shaft_segment,6*shaft_segment);
    %%%计算各轴段的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵
    for k=1:shaft_segment
        if k==1 %第1个截面
            %弯曲振动刚度
            I_section=pi*(shaft_segment_single(k,3)^4-shaft_segment_single(k,4)^4)/64;
            L_shaft=shaft_segment_single(k,2);
            E=shaft_segment_single(k,5);
            possion=shaft_segment_single(k,6);
            a_factor=shaft_segment_single(k,7);
            G=E/(2*(1+possion));
            area=(pi/4)*(shaft_segment_single(k,3)^2-shaft_segment_single(k,4)^2);
            if a_factor~=0
                f_sheer=6*E*I_section/(a_factor*G*area*L_shaft^2);
            else
                f_sheer=0;
            end
            beta_1=12*E*I_section/(L_shaft^3*(1+2*f_sheer));
            beta_2=0.5*L_shaft*beta_1;
            beta_3=L_shaft^2*(1-f_sheer)*beta_1/6;
            shaft_single_stiffness_local(6*k-5,6*k-5)=beta_1;
            shaft_single_stiffness_local(6*k-5,6*k-4)=beta_2;
            shaft_single_stiffness_local(6*k-5,6*k+1)=-beta_1;
            shaft_single_stiffness_local(6*k-5,6*k+2)=beta_2;
            shaft_single_stiffness_local(6*k-4,6*k-5)=beta_2;
            shaft_single_stiffness_local(6*k-4,6*k-4)=L_shaft*beta_2-beta_3;
            shaft_single_stiffness_local(6*k-4,6*k+1)=-beta_2;
            shaft_single_stiffness_local(6*k-4,6*k+2)=beta_3;
            shaft_single_stiffness_local(6*k-2,6*k-2)=beta_1;
            shaft_single_stiffness_local(6*k-2,6*k-1)=-beta_2;
            shaft_single_stiffness_local(6*k-2,6*k+4)=-beta_1;
            shaft_single_stiffness_local(6*k-2,6*k+5)=-beta_2;
            shaft_single_stiffness_local(6*k-1,6*k-2)=-beta_2;
            shaft_single_stiffness_local(6*k-1,6*k-1)=L_shaft*beta_2-beta_3;
            shaft_single_stiffness_local(6*k-1,6*k+4)=beta_2;
            shaft_single_stiffness_local(6*k-1,6*k+5)=beta_3;
            %轴向振动刚度
            axial_k=E*area/L_shaft;
            shaft_single_stiffness_local(6*k-3,6*k-3)=axial_k;
            shaft_single_stiffness_local(6*k-3,6*k+3)=-axial_k;
            %扭转振动刚度
            Ip=pi*(shaft_segment_single(k,3)^4-shaft_segment_single(k,4)^4)/32;
            stiffness_twist=G*Ip/L_shaft;
            shaft_single_stiffness_local(6*k,6*k)=stiffness_twist;
            shaft_single_stiffness_local(6*k,6*k+6)=-1*stiffness_twist;
        elseif k==shaft_segment %最后一个截面
            %弯曲振动刚度
            I_section=pi*(shaft_segment_single(k-1,3)^4-shaft_segment_single(k-1,4)^4)/64;
            L_shaft=shaft_segment_single(k-1,2);
            E=shaft_segment_single(k-1,5);
            possion=shaft_segment_single(k-1,6);
            a_factor=shaft_segment_single(k-1,7);
            G=E/(2*(1+possion));
            area=(pi/4)*(shaft_segment_single(k-1,3)^2-shaft_segment_single(k-1,4)^2);
            if a_factor~=0
                f_sheer=6*E*I_section/(a_factor*G*area*L_shaft^2);
            else
                f_sheer=0;
            end
            beta_1=12*E*I_section/(L_shaft^3*(1+2*f_sheer));
            beta_2=0.5*L_shaft*beta_1;
            beta_3=L_shaft^2*(1-f_sheer)*beta_1/6;
            shaft_single_stiffness_local(6*k-5,6*k-11)=-beta_1;
            shaft_single_stiffness_local(6*k-5,6*k-10)=-beta_2;
            shaft_single_stiffness_local(6*k-5,6*k-5)=beta_1;
            shaft_single_stiffness_local(6*k-5,6*k-4)=-beta_2;
            shaft_single_stiffness_local(6*k-4,6*k-11)=beta_2;
            shaft_single_stiffness_local(6*k-4,6*k-10)=beta_3;
            shaft_single_stiffness_local(6*k-4,6*k-5)=-beta_2;
            shaft_single_stiffness_local(6*k-4,6*k-4)=L_shaft*beta_2-beta_3;
            shaft_single_stiffness_local(6*k-2,6*k-8)=-beta_1;
            shaft_single_stiffness_local(6*k-2,6*k-7)=beta_2;
            shaft_single_stiffness_local(6*k-2,6*k-2)=beta_1;
            shaft_single_stiffness_local(6*k-2,6*k-1)=beta_2;
            shaft_single_stiffness_local(6*k-1,6*k-8)=-beta_2;
            shaft_single_stiffness_local(6*k-1,6*k-7)=beta_3;
            shaft_single_stiffness_local(6*k-1,6*k-2)=beta_2;
            shaft_single_stiffness_local(6*k-1,6*k-1)=L_shaft*beta_2-beta_3;
            %轴向振动刚度
            axial_k=E*area/L_shaft;
            shaft_single_stiffness_local(6*k-3,6*k-9)=-axial_k;
            shaft_single_stiffness_local(6*k-3,6*k-3)=axial_k;
            %扭转振动刚度
            Ip=pi*(shaft_segment_single(k-1,3)^4-shaft_segment_single(k-1,4)^4)/32;
            stiffness_twist=G*Ip/L_shaft;
            shaft_single_stiffness_local(6*k,6*k-6)=-stiffness_twist;
            shaft_single_stiffness_local(6*k,6*k)=stiffness_twist;
        else  %中间截面
            %弯曲振动刚度
            I_section_1=pi*(shaft_segment_single(k-1,3)^4-shaft_segment_single(k-1,4)^4)/64;
            L_shaft_1=shaft_segment_single(k-1,2);
            E_1=shaft_segment_single(k-1,5);
            possion_1=shaft_segment_single(k-1,6);
            a_factor_1=shaft_segment_single(k-1,7);
            G_1=E_1/(2*(1+possion_1));
            area_1=(pi/4)*(shaft_segment_single(k-1,3)^2-shaft_segment_single(k-1,4)^2);
            if a_factor_1~=0
                f_sheer_1=6*E_1*I_section_1/(a_factor_1*G_1*area_1*L_shaft_1^2);
            else
                f_sheer_1=0;
            end
            beta_1_1=12*E_1*I_section_1/(L_shaft_1^3*(1+2*f_sheer_1));
            beta_2_1=0.5*L_shaft_1*beta_1_1;
            beta_3_1=L_shaft_1^2*(1-f_sheer_1)*beta_1_1/6;
            I_section_2=pi*(shaft_segment_single(k,3)^4-shaft_segment_single(k,4)^4)/64;
            L_shaft_2=shaft_segment_single(k,2);
            E_2=shaft_segment_single(k,5);
            possion_2=shaft_segment_single(k,6);
            a_factor_2=shaft_segment_single(k,7);
            G_2=E_2/(2*(1+possion_2));
            area_2=(pi/4)*(shaft_segment_single(k,3)^2-shaft_segment_single(k,4)^2);
            if a_factor_2~=0
                f_sheer_2=6*E_2*I_section_2/(a_factor_2*G_2*area_2*L_shaft_2^2);
            else
                f_sheer_2=0;
            end
            beta_1_2=12*E_2*I_section_2/(L_shaft_2^3*(1+2*f_sheer_2));
            beta_2_2=0.5*L_shaft_2*beta_1_2;
            beta_3_2=L_shaft_2^2*(1-f_sheer_2)*beta_1_2/6;
            shaft_single_stiffness_local(6*k-5,6*k-11)=-beta_1_1;
            shaft_single_stiffness_local(6*k-5,6*k-10)=-beta_2_1;
            shaft_single_stiffness_local(6*k-5,6*k-5)=beta_1_1+beta_1_2;
            shaft_single_stiffness_local(6*k-5,6*k-4)=-beta_2_1+beta_2_2;
            shaft_single_stiffness_local(6*k-5,6*k+1)=-beta_1_2;
            shaft_single_stiffness_local(6*k-5,6*k+2)=beta_2_2;
            shaft_single_stiffness_local(6*k-4,6*k-11)=beta_2_1;
            shaft_single_stiffness_local(6*k-4,6*k-10)=beta_3_1;
            shaft_single_stiffness_local(6*k-4,6*k-5)=-beta_2_1+beta_2_2;
            shaft_single_stiffness_local(6*k-4,6*k-4)=(L_shaft_1*beta_2_1-beta_3_1)+(L_shaft_2*beta_2_2-beta_3_2);
            shaft_single_stiffness_local(6*k-4,6*k+1)=-beta_2_2;
            shaft_single_stiffness_local(6*k-4,6*k+2)=beta_3_2;
            shaft_single_stiffness_local(6*k-2,6*k-8)=-beta_1_1;
            shaft_single_stiffness_local(6*k-2,6*k-7)=beta_2_1;
            shaft_single_stiffness_local(6*k-2,6*k-2)=beta_1_1+beta_1_2;
            shaft_single_stiffness_local(6*k-2,6*k-1)=beta_2_1-beta_2_2;
            shaft_single_stiffness_local(6*k-2,6*k+4)=-beta_1_2;
            shaft_single_stiffness_local(6*k-2,6*k+5)=-beta_2_2;
            shaft_single_stiffness_local(6*k-1,6*k-8)=-beta_2_1;
            shaft_single_stiffness_local(6*k-1,6*k-7)=beta_3_1;
            shaft_single_stiffness_local(6*k-1,6*k-2)=beta_2_1-beta_2_2;
            shaft_single_stiffness_local(6*k-1,6*k-1)=(L_shaft_1*beta_2_1-beta_3_1)+(L_shaft_2*beta_2_2-beta_3_2);
            shaft_single_stiffness_local(6*k-1,6*k+4)=beta_2_2;
            shaft_single_stiffness_local(6*k-1,6*k+5)=beta_3_2;
            %轴向振动刚度
            axial_k_1=E_1*area_1/L_shaft_1;
            axial_k_2=E_2*area_2/L_shaft_2;
            shaft_single_stiffness_local(6*k-3,6*k-9)=-axial_k_1;
            shaft_single_stiffness_local(6*k-3,6*k-3)=axial_k_1+axial_k_2;
            shaft_single_stiffness_local(6*k-3,6*k+3)=-axial_k_2;
            %扭转振动刚度
            Ip_1=pi*(shaft_segment_single(k-1,3)^4-shaft_segment_single(k-1,4)^4)/32;
            stiffness_twist_1=G_1*Ip_1/L_shaft_1;
            Ip_2=pi*(shaft_segment_single(k,3)^4-shaft_segment_single(k,4)^4)/32;
            stiffness_twist_2=G_2*Ip_2/L_shaft_2;
            shaft_single_stiffness_local(6*k,6*k-6)=-1*stiffness_twist_1;
            shaft_single_stiffness_local(6*k,6*k)=stiffness_twist_1+stiffness_twist_2;
            shaft_single_stiffness_local(6*k,6*k+6)=-1*stiffness_twist_2;
        end
    end
    %%%
    %%%单个轴的弯曲振动、轴向振动、扭转振动的临时刚度矩阵（shaft_single_stiffness_temp）转化成单个轴的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，（shaft_single_stiffness）
    %单个轴的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，shaft_single_stiffness，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    %单个轴的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，shaft_single_stiffness_local，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness_local*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    %定义单个轴的弯曲振动、轴向振动、扭转振动的临时刚度矩阵，shaft_single_stiffness_temp
    %调整行
    shaft_single_stiffness_temp=zeros(6*shaft_segment,6*shaft_segment);
    for ii=1:shaft_segment  %序号为order_shaft轴的节点数，shaft_segment
        for jj=1:shaft_segment  %序号为order_shaft轴的节点数，shaft_segment
            %调整行
            shaft_single_stiffness_temp(6*(ii-1)+1,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+1,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+2,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+4,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+3,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+3,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+4,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+5,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+5,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+2,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+6,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+6,6*(jj-1)+1:6*jj);
            %调整列
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+1)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+1);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+2)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+4);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+3)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+3);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+4)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+5);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+5)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+2);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+6)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+6);
        end
    end
    %%%
    %%%附加螺母/丝杠/滚柱的轮缘轴对地的刚度
    addition_stiffness=zeros(6,6);
    addition_stiffness(1,1)=10^3;
    addition_stiffness(2,2)=10^3;
    addition_stiffness(3,3)=10^3;
    addition_stiffness(4,4)=10^5;
    addition_stiffness(5,5)=10^5;
    addition_stiffness(6,6)=10^5;
    shaft_single_stiffness(1:6,1:6)=addition_stiffness;
    %%%记录行星滚柱丝杠的弯曲振动、轴向振动、扭转振动的局部坐标系的刚度矩阵，roller_screw_stiffness，[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    roller_screw_stiffness(dd+1:dd+6*shaft_segment,1)=order_shaft; %螺母/丝杠/滚柱的缘轮轴序号，order_shaft
    roller_screw_stiffness(dd+1:dd+6*shaft_segment,2:1+6*shaft_segment)=shaft_single_stiffness;
    %%%更新计数器，dd
    dd=dd+6*shaft_segment;
end
end
%%%程序结束