%%%�������ǹ���˿�ܵ��غɼ��Ӵ�Ӧ���ֲ�
function [potential_total,Fn_nut,Fn_screw,flexible_inner,flexible_outer,B_inner,B_outer,half_width_a_inner,half_width_b_inner,stress_inner,half_width_a_outer,half_width_b_outer,stress_outer,ratio_nut,ratio_screw,distribution_nut,distribution_screw,delta_outer,delta_inner,S_load_nut,S_load_screw]=calulate_load_distribution_roller_screw(order_roller_screw,order_load,roller_screw_initial,type_load,norm_roller_screw,mesh_outer_matrix,mesh_inner_matrix,modify_nut,modify_screw)
%%%�������
%���ǹ���˿���ܵ����ܣ�potential_total
%��ĸ�ĽӴ��㷨���غɵ�������Fn_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
%˿�ܵĽӴ��㷨���غɵ�������Fn_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
%�����Ƶ��������(mm/N)��flexible_inner
%�����Ƶ��������(mm/N)��flexible_outer
%������ѹ�����ܸ���ֵ��B_inner
%������ѹ�����ܸ���ֵ��B_outer
%�����ϵĽӴ�������(mm)��half_width_a_inner
%�����ϵĽӴ��̰���(mm)��half_width_b_inner
%�����ϽӴ�Ӧ��(Mpa)��stress_inner
%�����ϵĽӴ�������(mm)��half_width_a_outer
%�����ϵĽӴ��̰���(mm)��half_width_b_outer
%�����ϽӴ�Ӧ��(Mpa)��stress_outer
%��ĸ�������������ratio_nut
%˿���������������ratio_screw
%��ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
%��ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
%�����ϽӴ�������(mm)��delta_inner
%�����ϽӴ�������(mm)��delta_outer
%��ĸ�غɷֲ�����ֵ��S_load_nut
%˿���غɷֲ�����ֵ��S_load_screw
%%%��������
%���ǹ���˿����ţ�order_roller_screw
%������ţ�order_load
%���ǹ���˿�ܻ���������roller_screw_initial��[1 ���ǹ���˿����� 2 ���ǹ���˿�ܵ�����(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw 3 ��������,number_roller 4 ˿������ͷ��,N_thread_s 5 ��ĸ����ͷ��,N_thread_n 6 ��������ͷ��,N_thread_r
%7 ˿����������(-1��ʾ������1��ʾ����),direct_screw 8 ��ĸ��������(-1��ʾ������1��ʾ����),direct_nut 9 ������������(-1��ʾ������1��ʾ����),direct_roller 10 ˿����������(rad)��rise_angle_s 11 ��ĸ��������(rad)��rise_angle_n 12 ������������(rad)��rise_angle_r
%13 ��1��������X��н�(rad)��angle_roller 14 �����Ƶ���Ħ��ϵ����f_inner 15 �����Ƶ���Ħ��ϵ����f_outer 16 �ݾ�(mm)��pitch_screw 17 ���ͽ�(rad)��angle_screw 18 ��������Բ���뾶(mm)��radius_roller 19 ��������Բ�Ľ�������ϵX��ֵ(mm)��X_center_r 20 ��������Բ�Ľ�������ϵY��ֵ(mm)��Y_center_r
%21 ˿����������ϵ����cutter_top_s 22 ˿����������ϵ����cutter_bottom_s 23 ��ĸ��������ϵ����cutter_top_n 24 ��ĸ��������ϵ����cutter_bottom_n 25 ������������ϵ����cutter_top_r 26 ������������ϵ����cutter_bottom_r 27 ˿�����ư���������(mm)��reduce_s 28 ��ĸ���ư���������(mm)��reduce_n 29 �������ư���������(mm)��reduce_r 30 ��
%31 ˿�ܵ����Ƴݶ���(mm)��addendum_s 32 ˿�ܵ����Ƴݸ���(mm)��dedendum_s 33 ��ĸ�����Ƴݶ���(mm)��addendum_n 34 ��ĸ�����Ƴݸ���(mm)��dedendum_n 35 ���������Ƴݶ���(mm)��addendum_r 36 ���������Ƴݸ���(mm)��dedendum_r
%37 ˿��ʵ�ʶ���(mm)��D_top_s 38 ˿��ʵ�ʸ���(mm)��D_bottom_s 39 ��ĸʵ�ʶ���(mm)��D_top_n
%40 ��ĸʵ�ʸ���(mm)��D_bottom_n 41 ����ʵ�ʶ���(mm)��D_top_r 42 ����ʵ�ʸ���(mm)��D_bottom_r
%43 ˿���о�(mm),D_pitch_s ,44 ��ĸ�о�(mm),D_pitch_n, 45 �����о�(mm)��D_pitch_r
%46 ˿���᳤��(mm)��length_shaft_s 47 ��ĸ�᳤��(mm)��length_shaft_n 48 �����᳤��(mm)��length_shaft_r 49 ˿�����⾶(mm)��diameter_outer_s 50 ��ĸ���⾶(mm)��diameter_outer_n 51 �������⾶(mm)��diameter_outer_r 52 ˿�����ھ�(mm)��diameter_inner_s 53 ��ĸ���ھ�(mm)��diameter_inner_n 54 �������ھ�(mm)��diameter_inner_r
%55 ˿�����Ƴ���(mm)��length_thread_s 56 ��ĸ���Ƴ���(mm)��length_thread_n 57 �������Ƴ���(mm)��length_thread_r 58 �����ϵ���Ч���Ƴ���(mm)��length_mesh_inner 59 �����ϵ���Ч���Ƴ���(mm)��length_mesh_outer 60 ��
%61 ����ֿ��(mm)��width_gear_left 62 �ҳ��ֿ��(mm)��width_gear_left 63 �󱣳ּܿ��(mm)��width_carrier_left 64 �󱣳ּܿ��(mm)��width_carrier_right
%65 ��ĸ����������˿������˵�λ��(mm)��delta_n_s 66 ��������������˿������˵�λ��(mm)��delta_r_s 67 ˿��������������˿������˵�λ��(mm),delta_thread_s 68 ��ĸ��������������ĸ����˵�λ��(mm),delta_thread_n 69 ���������������ڹ�������˵�λ��(mm),delta_thread_r 70 ����Ԥ���غ�(N)��preload_axial
%71 ˿������ܶ�(t/mm^-9)��density_s 72 ��ĸ����ܶ�(t/mm^-9)��density_n 73 ��������ܶ�(t/mm^-9)��density_r 74 ˿�ܵ���ģ��(Mpa)��E_s 75 ��ĸ����ģ��(Mpa)��E_n 76 ��������ģ��(Mpa)��E_r 77 ˿�ܲ��ɱȣ�possion_s 78 ��ĸ���ɱȣ�possion_n 79 �������ɱȣ�possion_r
%80 ������������ڹ��������λ��(mm)��delta_gear_left 81 �ҳ����������ڹ��������λ��(mm)��delta_gear_right 82 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_left 83 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_right
%84 �󱣳ּ��⾶(mm)��outer_carrier_left 85 �󱣳ּ��ھ�(mm)��inner_carrier_left 86 �ұ��ּ��⾶(mm)��outer_carrier_right 87 �ұ��ּ��ھ�(mm)��inner_carrier_right 88 ���ǹ���˿�Ƿ�ָ��Ч��(0��ʾ��ָ����1��ʾָ��),sign_efficiency 89 ��������Ч��(��Ԥ���غ�)��advance_efficiency 90 ��������Ч��(��Ԥ���غ�)��reverse_efficiency
%91 ���ַ���ģ��(mm)��m_n 92 �ڳ�Ȧ������z_r 93 ̫���ֳ�����z_s 94 �����ֳ�����z_p 95 ѹ����(rad)��press_angle 96 ����ѹ����(rad)��tran_press_angle 97 ������(rad)��helix_angle 98 �ڳ�Ȧ�����λϵ����n_r 99 ̫���ַ����λϵ����n_s 100 �����ַ����λϵ����n_p
%101 �������ľ�(mm)��work_center 102 ������ϵ�����Ͻǣ�mesh_angle(rad) 103 ������ϵ�����Ͻǣ�mesh_angle(rad) 104 �ڳ�Ȧ�ݶ�Բ��tran_ad_dia_r��mm�� 105 ̫���ֳݶ�Բ��tran_ad_dia_s��mm�� 106 �����ֳݶ�Բ��tran_ad_dia_p��mm�� 107 �ڳ�Ȧ�ݸ�Բ��tran_de_dia_r��mm�� 108 ̫���ֳݸ�Բ��tran_de_dia_s��mm�� 109 �����ֳݸ�Բ��tran_de_dia_p��mm��]
%110 �ڳ�Ȧ�Ľ�Բ��mm����tran_pitch_dia_r 111 ̫���ֵĽ�Բ��mm����tran_pitch_dia_s 112 �����ֵĽ�Բ��mm����tran_pitch_dia_p 113-120 ��
%121 ��ĸ������(mm)��width_web_n 122 ��ĸ�������ľ���ĸ����˾���(mm)��delta_web_n 123 ˿���������(mm)��width_left_web_s 124 ˿����������ľ�˿������˾���(mm)��delta_left_web_s 125 ˿���ҷ�����(mm)��width_right_web_s 126 ˿���ҷ������ľ�˿������˾���(mm)��delta_right_web_s 127-130 ��
%131 ��ĸ��Ե�����,shaft_self_nut 132 ˿����Ե����ţ�shaft_self_screw 133 ������Ե����ţ�shaft_self_roller,134-140 ��
%141 �����ֵ��߳ݶ���ϵ����142 �����ֵ��߶�϶ϵ����143 �����ֳݶ���������144 �ڳ�Ȧ���߳ݶ���ϵ����145 �ڳ�Ȧ���߶�϶ϵ����146 �ڳ�Ȧ�ݶ�������,147 ̫���ֵ��߳ݶ���ϵ����148 ̫���ֵ��߶�϶ϵ����149 ̫���ֳݶ�������
%150-160 �� 161 ��Ч�����������������ĸ����˵�λ��(mm)��thread_inner_nut 162 ��Ч��������������ڹ�������˵�λ��(mm)��thread_inner_roller 163 ��Ч���������������˿������˵�λ��(mm)��thread_outer_screw 164 ��Ч��������������ڹ�������˵�λ��(mm)��thread_outer_roller]
%���ǹ���˿�����ط�ʽ��type_load��1��ʾͬ�����أ�2��ʾ�������
%���ǹ���˿��������/�����Ƶķ����غ���ؾ���norm_roller_screw��[1 ������� 2 ���ǹ���˿����� 3 ˿�ܹ���(kW) 4 ��ĸ����(kW) 5  ��Ԥ��Ч��
%6 ˿������� 7 ˿�ܹ�����(-1��ʾ��������1��ʾ������) 8 ˿�ܹ���ѹ����(rad) 9 ˿�����ظ��� 10 ˿�ܷ����غ�(N) 11 ˿�ܵķ���ʸ��X����� 12 ˿�ܵķ���ʸ��Y����� 13 ˿�ܵķ���ʸ��Z�����
%14 ��ĸ����� 15 ��ĸ������(-1��ʾ��������1��ʾ������) 16 ��ĸ����ѹ����(rad) 17 ��ĸ���ظ��� 18 ��ĸ�����غ�(N) 19 ��ĸ�ķ���ʸ��X����� 20 ��ĸ�ķ���ʸ��Y����� 21 ��ĸ�ķ���ʸ��Z�����
%22 ��ת������Ԥ�����ķ����غ� 23 ������ʽ(1��ʾ����������2��ʾ��������) 24 ֱ�߻������ٶ�(mm/s),velocity_linear 25 ��ת������ת�ٶ�(rpm),speed_turn 26 �������ת���ٶ�(rpm),speed_roller 27 ����ʱ��(s),time 28 ѭ��������number_cycle 29 ���ǹ���˿�ܵ�����(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw]
%���ǹ���˿������������λ�þ���mesh_outer_matrix��[1 ���ǹ���˿����� 2 ˿�ܹ�����(-1��ʾ��������1��ʾ������) 3 ����������(-1��ʾ��������1��ʾ������) 4 �����ϵ�ȫ������ϵX������(mm) 5 �����ϵ�ȫ������ϵY������(mm) 6 ˿�������ϵ�н�(rad) 7 ˿�������ϵ�뾶(mm) 8 ���������ϼн�(rad) 9 �����ϵ�뾶(mm) 10 �����������϶(mm)]
%���ǹ���˿������������λ�þ���mesh_inner_matrix��[1 ���ǹ���˿����� 2 ��ĸ������(-1��ʾ��������1��ʾ������) 3 ����������(-1��ʾ��������1��ʾ������) 4 �����ϵ�ȫ������ϵX������(mm) 5 �����ϵ�ȫ������ϵY������(mm) 6 ��ĸ�����ϵ�н�(rad) 7 ��ĸ�����ϵ�뾶(mm) 8 ���������ϼн�(rad) 9 �����ϵ�뾶(mm) 10 �����������϶(mm)]
%��ĸ����������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(um)��modify_nut
%˿�ܲ���������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(um)��modify_screw
%%%����ʼ
%%%��¼���ȽӴ�ϵ����H
H=[0,0.02363,0.02363,0.0002791;0.05,0.02444,0.02286,0.000279;0.1,0.02529,0.02212,0.0002785;0.15,0.0262,0.02142,0.0002777;0.2,0.02717,0.02074,0.0002766;0.25,0.02821,0.02008,0.0002751;0.3,0.02934,0.01943,0.0002733;0.35,0.03057,0.0188,0.0002711;0.4,0.03192,0.01818,0.0002685;0.45,0.03342,0.01756,0.0002654;0.5,0.03511,0.01694,0.0002617;0.55,0.03703,0.01632,0.0002574;0.6,0.03924,0.01569,0.0002525;0.62,0.04023,0.01544,0.0002502;0.64,0.04130,0.01517,0.0002479;0.66,0.04244,0.01491,0.0002453;0.68,0.04367,0.01464,0.0002426;0.7,0.04502,0.01437,0.0002397;0.71,0.04573,0.01423,0.0002381;0.72,0.04649,0.01408,0.0002365;0.73,0.04726,0.01394,0.0002349;0.74,0.04809,0.01380,0.0002331;0.75,0.04896,0.01365,0.0002313;0.76,0.04988,0.01350,0.0002294;0.77,0.05086,0.01334,0.0002275;0.78,0.05189,0.01319,0.0002254;0.79,0.05299,0.01303,0.0002233;0.8,0.05416,0.01286,0.000221;0.805,0.05477,0.01278,0.0002198;0.81,0.05540,0.01269,0.0002186;0.815,0.05606,0.01261,0.0002174;0.82,0.05674,0.01252,0.0002162;0.825,0.05744,0.01244,0.0002149;0.83,0.05817,0.01235,0.0002136;0.835,0.05895,0.01225,0.0002122;0.840,0.05974,0.01216,0.0002108;0.845,0.06058,0.01207,0.0002093;0.85,0.06144,0.01197,0.0002079;0.855,0.06236,0.01187,0.0002063;0.86,0.06330,0.01178,0.0002047;0.865,0.06430,0.01167,0.0002031;0.87,0.06537,0.01157,0.0002014;0.875,0.06648,0.01146,0.0001996;0.88,0.06767,0.01135,0.0001979;0.885,0.06890,0.01124,0.0001959;0.89,0.07017,0.01112,0.000194;0.895,0.07161,0.01101,0.0001919;0.9,0.07305,0.01089,0.0001898;0.905,0.07469,0.01076,0.0001875;0.91,0.07641,0.01063,0.0001852;0.915,0.07823,0.01050,0.0001828;0.92,0.08023,0.01036,0.0001802;0.925,0.08247,0.01021,0.0001775;0.93,0.08480,0.01006,0.0001747;0.935,0.08743,0.009898,0.0001716;0.94,0.09035,0.009729,0.0001684;0.945,0.09362,0.009550,0.0001649;0.95,0.09733,0.009359,0.0001611;0.955,0.1016,0.009153,0.000157;0.96,0.1066,0.008930,0.0001525;0.965,0.1124,0.008690,0.0001476;0.970,0.1197,0.008414,0.0001419;0.975,0.1284,0.008118,0.0001357;0.98,0.1404,0.007757,0.0001281;0.985,0.1573,0.007323,0.0001189;0.99,0.1831,0.006784,0.0001074;0.995,0.2398,0.005923,0.00008911];
%%%�������ǹ���˿�ܵ��غɼ�Ӧ���ֲ�
if sum(roller_screw_initial(:,1)==order_roller_screw)>0 %���ǹ���˿����ţ�order_roller_screw
    %%%������ĸ����������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_nut
    modify_nut=10^(-3)*modify_nut;
    modify_nut=modify_nut-min(modify_nut);
    %%%����˿�ܲ���������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_screw
    modify_screw=10^(-3)*modify_screw;
    modify_screw=modify_screw-min(modify_screw);
    %%%��¼���Ϊorder_roller_screw�����ǹ���˿�ܵĻ���������roller_screw_temp
    id_roller_screw=(roller_screw_initial(:,1)==order_roller_screw);
    roller_screw_temp=roller_screw_initial(id_roller_screw,:);
    %%%��¼���ǹ���˿�ܵ�(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw
    type_roller_screw=roller_screw_temp(1,2);
    %%%��¼���ǹ���˿�ܵĹ���������number_roller
    number_roller=roller_screw_temp(1,3);
    %%%��¼���ǹ���˿�ܵ��ݾ�(mm)��pitch_screw
    pitch_screw=roller_screw_temp(1,16);
    %%%��¼��������Բ���뾶(mm)��radius_roller
    radius_roller=roller_screw_temp(1,18);
    %%%��¼˿��ʵ�ʸ���(mm)��D_bottom_s
    D_bottom_s=roller_screw_temp(1,38);
    %%%��¼��ĸʵ�ʸ���(mm)��D_bottom_n
    D_bottom_n=roller_screw_temp(1,40);
    %%%��¼����ʵ�ʸ���(mm)��D_bottom_r
    D_bottom_r=roller_screw_temp(1,42);
    %%%��¼��ĸ���⾶(mm)��diameter_outer_n
    diameter_outer_n=roller_screw_temp(1,50);
    %%%��¼˿�����ھ�(mm)��diameter_inner_s
    diameter_inner_s=roller_screw_temp(1,52);
    %%%��¼�������ھ�(mm)��diameter_inner_r
    diameter_inner_r=roller_screw_temp(1,54);
    %%%��¼�����ϵ���Ч���Ƴ���(mm)��length_mesh_inner
    length_mesh_inner=roller_screw_temp(1,58);
    %%%��¼�����ϵ���Ч���Ƴ���(mm)��length_mesh_outer
    length_mesh_outer=roller_screw_temp(1,59);
    %%%��¼˿�ܵ���ģ��(Mpa)��E_s
    E_s=roller_screw_temp(1,74);
    %%%��¼��ĸ����ģ��(Mpa)��E_n
    E_n=roller_screw_temp(1,75);
    %%%��¼��������ģ��(Mpa)��E_r
    E_r=roller_screw_temp(1,76);
    %%%��¼˿�ܲ��ɱȣ�possion_s
    possion_s=roller_screw_temp(1,77);
    %%%��¼��ĸ���ɱȣ�possion_n
    possion_n=roller_screw_temp(1,78);
    %%%��¼�������ɱȣ�possion_r
    possion_r=roller_screw_temp(1,79);
    %%%����˿���ۺ�ģ����E0_s
    E0_s=E_s/(1-possion_s^2);
    %%%������ĸ�ۺ�ģ����E0_n
    E0_n=E_n/(1-possion_n^2);
    %%%��������ۺ�ģ����E0_r
    E0_r=E_r/(1-possion_r^2);
    %%%������ĸ�����(mm^2)��area_n
    area_n=pi*(diameter_outer_n^2-D_bottom_n^2)/4;
    %%%����˿�ܽ����(mm^2)��area_s
    area_s=pi*(D_bottom_s^2-diameter_inner_s^2)/4;
    %%%������������(mm^2)��area_r
    area_r=pi*(D_bottom_r^2-diameter_inner_r^2)/4;
    %%%
    %%%��¼���ǹ���˿�ܵ����Ϊorder_load��������˿�ܹ����棬face_screw
    %���ǹ���˿��������/�����Ƶķ����غ���ؾ���norm_roller_screw��[1 ������� 2 ���ǹ���˿����� 3 ˿�ܹ���(kW) 4 ��ĸ����(kW) 5  ��Ԥ��Ч��
    %6 ˿������� 7 ˿�ܹ�����(-1��ʾ��������1��ʾ������) 8 ˿�ܹ���ѹ����(rad) 9 ˿�����ظ��� 10 ˿�ܷ����غ�(N) 11 ˿�ܵķ���ʸ��X����� 12 ˿�ܵķ���ʸ��Y����� 13 ˿�ܵķ���ʸ��Z�����
    %14 ��ĸ����� 15 ��ĸ������(-1��ʾ��������1��ʾ������) 16 ��ĸ����ѹ����(rad) 17 ��ĸ���ظ��� 18 ��ĸ�����غ�(N) 19 ��ĸ�ķ���ʸ��X����� 20 ��ĸ�ķ���ʸ��Y����� 21 ��ĸ�ķ���ʸ��Z�����
    %22 ��ת������Ԥ�����ķ����غ� 23 ������ʽ(1��ʾ����������2��ʾ��������) 24 ֱ�߻������ٶ�(mm/s),velocity_linear 25 ��ת������ת�ٶ�(rpm),speed_turn 26 �������ת���ٶ�(rpm),speed_roller 27 ����ʱ��(s),time 28 ѭ��������number_cycle 29 ���ǹ���˿�ܵ�����(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw]
    id_norm_1=(norm_roller_screw(:,1)==order_load); %���Ϊorder_load��������ţ�order_load
    id_norm_2=(norm_roller_screw(:,2)==order_roller_screw); %���ǹ���˿�ܵ���ţ�order_roller_screw
    id_norm=(id_norm_1+id_norm_2==2);
    face_screw=norm_roller_screw(id_norm,7);
    %%%��¼˿�ܹ���ѹ����(rad)��press_angle_screw
    press_angle_screw=abs(norm_roller_screw(id_norm,8));
    %%%��¼˿�ܷ����غ�(N)��force_screw
    force_screw=abs(norm_roller_screw(id_norm,10));
    %%%��¼˿�ܵ����ط���ʸ����vector_norm_screw
    vector_norm_screw=norm_roller_screw(id_norm,11:13);
    %%%����˿���������������ratio_screw
    ratio_screw=abs(vector_norm_screw(1,3))/norm(vector_norm_screw);
    %%%��¼���ǹ���˿�ܵ����Ϊorder_load����������ĸ�����棬face_nut
    face_nut=norm_roller_screw(id_norm,15);
    %%%��¼��ĸ����ѹ����(rad)��press_angle_nut
    press_angle_nut=abs(norm_roller_screw(id_norm,16));
    %%%��¼��ĸ�����غ�(N)��force_nut
    force_nut=abs(norm_roller_screw(id_norm,18));
    %%%��¼��ĸ�����ط���ʸ����vector_norm_nut
    vector_norm_nut=norm_roller_screw(id_norm,19:21);
    %%%������ĸ�������������ratio_nut
    ratio_nut=abs(vector_norm_nut(1,3))/norm(vector_norm_nut);
    %%%
    %%%��¼˿�������ϵ�뾶(mm)��radius_screw
    %���ǹ���˿������������λ�þ���mesh_outer_matrix��[1 ���ǹ���˿����� 2 ˿�ܹ�����(-1��ʾ��������1��ʾ������) 3 ����������(-1��ʾ��������1��ʾ������) 4 �����ϵ�ȫ������ϵX������(mm) 5 �����ϵ�ȫ������ϵY������(mm) 6 ˿�������ϵ�н�(rad) 7 ˿�������ϵ�뾶(mm) 8 ���������ϼн�(rad) 9 �����ϵ�뾶(mm) 10 �����������϶(mm)]
    id_mesh_outer_1=(mesh_outer_matrix(:,1)==order_roller_screw); %���ǹ���˿�ܵ���ţ�order_roller_screw
    id_mesh_outer_2=(mesh_outer_matrix(:,2)==face_screw); %˿�ܹ����棬face_screw
    id_mesh_outer=(id_mesh_outer_1+id_mesh_outer_2==2);
    radius_screw=mesh_outer_matrix(id_mesh_outer,7);
    %%%��¼���������ϰ뾶(mm)��radius_outer
    radius_outer=mesh_outer_matrix(id_mesh_outer,9);
    %%%
    %%%��¼��ĸ�����ϵ�뾶(mm)��radius_nut
    %���ǹ���˿������������λ�þ���mesh_inner_matrix��[1 ���ǹ���˿����� 2 ��ĸ������(-1��ʾ��������1��ʾ������) 3 ����������(-1��ʾ��������1��ʾ������) 4 �����ϵ�ȫ������ϵX������(mm) 5 �����ϵ�ȫ������ϵY������(mm) 6 ��ĸ�����ϵ�н�(rad) 7 ��ĸ�����ϵ�뾶(mm) 8 ���������ϼн�(rad) 9 �����ϵ�뾶(mm) 10 �����������϶(mm)]
    id_mesh_inner_1=(mesh_inner_matrix(:,1)==order_roller_screw); %���ǹ���˿�ܵ���ţ�order_roller_screw
    id_mesh_inner_2=(mesh_inner_matrix(:,2)==face_nut); %��ĸ�����棬face_nut
    id_mesh_inner=(id_mesh_inner_1+id_mesh_inner_2==2);
    radius_nut=mesh_inner_matrix(id_mesh_inner,7);
    %%%��¼���������ϰ뾶(mm)��radius_inner
    radius_inner=mesh_inner_matrix(id_mesh_inner,9);
    %%%
    %%%���������Ϲ�����1������(1/mm)��p_outer_1
    %��������Բ���뾶(mm)��radius_roller
    p_outer_1=1/radius_roller;
    %%%���������Ϲ�����2������(1/mm),p_outer_2
    %���������ϰ뾶(mm)��radius_outer
    %˿�ܹ���ѹ����(rad)��press_angle_screw
    p_outer_2=cos(pi/2-press_angle_screw)/radius_outer;
    %%%����������˿�ܵ�1������(1/mm),p_screw_1
    p_screw_1=0;
    %%%����������˿�ܵ�2������(1/mm)��p_screw_2
    %˿�ܹ���ѹ����(rad)��press_angle_screw
    p_screw_2=cos(pi/2-press_angle_screw)/radius_screw;
    %%%��������������֮��(1/mm)��sum_p_outer
    sum_p_outer=p_outer_1+p_outer_2+p_screw_1+p_screw_2;
    %%%���������������ʺ�����F_p_outer
    F_p_outer=(abs(p_outer_1-p_outer_2)+abs(p_screw_1-p_screw_2))/sum_p_outer;
    %%%�����������ۺ�ģ��Ӱ��ϵ����E0_outer
    E0_outer=(1.137*10^5*(1/E0_s+1/E0_r))^(1/3);
    %%%����������ѹ�����ܸ���ֵ��B_outer
    B_outer=0.25*(p_outer_1+p_outer_2)+(p_screw_1+p_screw_2)-abs(p_outer_1-p_outer_2)-abs(p_screw_1-p_screw_2);
    %%%
    %%%���������Ϲ�����1������(1/mm)��p_inner_1
    %��������Բ���뾶(mm)��radius_roller
    p_inner_1=1/radius_roller;
    %%%���������Ϲ�����2������(1/mm),p_inner_2
    %���������ϰ뾶(mm)��radius_inner
    %��ĸ����ѹ����(rad)��press_angle_nut
    p_inner_2=cos(pi/2-press_angle_nut)/radius_inner;
    %%%������������ĸ��1������(1/mm),p_nut_1
    p_nut_1=0;
    %%%������������ĸ��2������(1/mm)��p_nut_2
    %��ĸ����ѹ����(rad)��press_angle_nut
    p_nut_2=-1*cos(pi/2-press_angle_nut)/radius_nut;
    %%%��������������֮��(1/mm)��sum_p_inner
    sum_p_inner=p_inner_1+p_inner_2+p_nut_1+p_nut_2;
    %%%���������������ʺ�����F_p_inner
    F_p_inner=(abs(p_inner_1-p_inner_2)+abs(p_nut_1-p_nut_2))/sum_p_inner;
    %%%�����������ۺ�ģ��Ӱ��ϵ����E0_inner
    E0_inner=(1.137*10^5*(1/E0_n+1/E0_r))^(1/3);
    %%%����������ѹ�����ܸ���ֵ��B_inner
    B_inner=0.25*(p_inner_1+p_inner_2)+(p_nut_1+p_nut_2)-abs(p_inner_1-p_inner_2)-abs(p_nut_1-p_nut_2);
    %%%
    %%%���������ϵĳ�����/�̰���/�Ӵ�����Ӱ��ϵ����C_a_outer/C_b_outer/C_delta_outer
    %���ȽӴ�ϵ����H
    C_a_outer=interp1(H(:,1),H(:,2),F_p_outer,'spline','extrap');
    C_b_outer=interp1(H(:,1),H(:,3),F_p_outer,'spline','extrap');
    C_delta_outer=interp1(H(:,1),H(:,4),F_p_outer,'spline','extrap');
    %%%
    %%%���������ϵĳ�����/�̰���/�Ӵ�����Ӱ��ϵ����C_a_inner/C_b_inner/C_delta_inner
    %���ȽӴ�ϵ����H
    C_a_inner=interp1(H(:,1),H(:,2),F_p_inner,'spline','extrap');
    C_b_inner=interp1(H(:,1),H(:,3),F_p_inner,'spline','extrap');
    C_delta_inner=interp1(H(:,1),H(:,4),F_p_inner,'spline','extrap');
    %%%
    %%%����������������������number
    number=1+fix(min(length_mesh_inner,length_mesh_outer)/pitch_screw);
    %%%�ж���ĸ����������ά���Ƿ���ȷ
    [row_modify,column_modify]=size(modify_nut);
    if (row_modify~=number)||(column_modify~=number_roller)
        h1=msgbox('��ĸ����������ά������ȷ��������ֹ');
        javaFrame=get(h1,'JavaFrame');
        javaFrame.setFigureIcon(javax.swing.ImageIcon('�ɻ�.jpg'));
        %%%�˳�����
        return;
    end
    %%%�ж�˿�ܲ���������ά���Ƿ���ȷ
    [row_modify,column_modify]=size(modify_screw);
    if (row_modify~=number)||(column_modify~=number_roller)
        h1=msgbox('˿�ܲ���������ά������ȷ��������ֹ');
        javaFrame=get(h1,'JavaFrame');
        javaFrame.setFigureIcon(javax.swing.ImageIcon('�ɻ�.jpg'));
        %%%�˳�����
        return;
    end
    %%%�����غ����õ㣬number_load
    number_load=number*number_roller;
    %%%�������ǹ���˿���ܵ������غ�(N)��force_axial
    if type_roller_screw==1 %���ǹ���˿�ܵ�(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw
        %��ĸ�����غ�(N)��force_nut
        %��ĸ�������������ratio_nut
        force_axial=abs(force_nut)*ratio_nut*number_load;
    elseif type_roller_screw==2 %���ǹ���˿�ܵ�(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw
        %˿�ܷ����غ�(N)��force_screw
        %%˿���������������ratio_screw
        force_axial=abs(force_screw)*ratio_screw*number_load;
    end
    %%%
    %%%�������ǹ���˿��������/�����Ƶ��������
    [flexible_inner,flexible_outer]=flexible_roller_screw(roller_screw_temp);
    %�����Ƶ��������(mm/N)��flexible_inner
    %�����Ƶ��������(mm/N)��flexible_outer
    %%%
    %%%������ǹ���˿�ܸ��Ӵ�������ط����غ�
    if type_load==1 %���ǹ���˿�����ط�ʽ��type_load��1��ʾͬ�����أ�2��ʾ�������
        %%%���ó�ʼֵ
        F0=force_nut*ones(2*number,number_roller);
        %%%������ĸ���㹤����϶(mm)��nut_temp
        nut_temp=zeros(number,number_roller);
        %%%����˿�ܲ��㹤����϶(mm)��screw_temp
        screw_temp=zeros(number,number_roller);
        %%%������ĸ�๤����϶����ֵ����(mm)��max_nut
        max_nut=zeros(number,number_roller);
        %%%����˿�ܲ๤����϶����ֵ����(mm)��max_screw
        max_screw=zeros(number,number_roller);
        %%%��������غɷ���ϵ����share
% % %         share=[0.16;0.21;0.21;0.21;0.21];
        share=(1/number_roller)*ones(number_roller,1);
        %%%����fsolve����㹤����϶�ķ����غɣ�Fn_refer
        options=optimoptions('fsolve','OptimalityTolerance',10^-6);
        [Fn_refer]=fsolve(@(Fn)load_roller_screw_1(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,nut_temp,screw_temp,share),F0,options);
        %%%���㿼��ʵ�ʹ�����϶�ĸ��Ӵ��㷨���غ�
        if (sum(sum(modify_nut))+sum(sum(modify_screw)))>0
            %%%������ĸ��/˿�ܲ๤����϶��ֵ(mm)��max_nut/max_screw
            for kk=1:number_roller %���ǹ���˿�ܵĹ���������number_roller
                for ii=2:number %������������������number
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
            %%%������ĸ��/˿�ܲ���������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_nut/modify_screw
            modify_nut=min(modify_nut,max_nut);
            modify_screw=min(modify_screw,max_screw);
            %%%��⿼�ǹ�����϶ʱ���ǹ���˿�ܸ��Ӵ�������ط����غ�
            [Fn,fval]=fsolve(@(Fn)load_roller_screw_1(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share),F0,options);
            if (sum(sum(abs(fval(2:number,1:number_roller)))>0.01)>0)||(sum(sum(abs(fval(2+number:2*number,1:number_roller)))>0.01)>0)
                h1=msgbox('���ǹ���˿�ܷ����غɼ�����ڲ�����');
                javaFrame=get(h1,'JavaFrame');
                javaFrame.setFigureIcon(javax.swing.ImageIcon('�ɻ�.jpg'));
            end
        else
            %%%����ʵ�ʹ�����϶�ĸ��Ӵ��㷨���غ�
            Fn=Fn_refer;
        end
        %%%�������ǹ���˿�ܸ��Ӵ�������ط����غ�
        Fn(:,:)=max(0,Fn(:,:));
        %%%��¼��ĸ�ĽӴ��㷨���غɵ�������Fn_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
        Fn_nut(1:number,1)=(1:number)';
        Fn_nut(1:number,2:1+number_roller)=Fn(1:number,1:number_roller);
        %%%��¼��ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
        distribution_nut(1:number,1)=(1:number)';
        distribution_nut(1:number,2:1+number_roller)=Fn_nut(1:number,2:1+number_roller)/(force_axial/number_load/ratio_nut);
        %%%˿�ܵĽӴ��㷨���غɵ�������Fn_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
        Fn_screw(1:number,1)=(1:number)';
        Fn_screw(1:number,2:1+number_roller)=Fn(1+number:2*number,1:number_roller);
        %%%��¼��ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
        distribution_screw(1:number,1)=(1:number)';
        distribution_screw(1:number,2:1+number_roller)=Fn_screw(1:number,2:1+number_roller)/(force_axial/number_load/ratio_screw);
    end
    %%%
    if type_load==2 %���ǹ���˿�����ط�ʽ��type_load��1��ʾͬ�����أ�2��ʾ�������
        %%%���ó�ʼֵ
        F0=force_nut*ones(2*number,number_roller);
        %%%������ĸ���㹤����϶(mm)��nut_temp
        nut_temp=zeros(number,number_roller);
        %%%����˿�ܲ��㹤����϶(mm)��screw_temp
        screw_temp=zeros(number,number_roller);
        %%%������ĸ�๤����϶����ֵ����(mm)��max_nut
        max_nut=zeros(number,number_roller);
        %%%����˿�ܲ๤����϶����ֵ����(mm)��max_screw
        max_screw=zeros(number,number_roller);
        %%%��������غɷ���ϵ����share
% % %         share=[0.16;0.21;0.21;0.21;0.21];
        share=(1/number_roller)*ones(number_roller,1);
        %%%����fsolve����㹤����϶�ķ����غɣ�Fn_refer
        options=optimoptions('fsolve','OptimalityTolerance',10^-6);
        [Fn_refer]=fsolve(@(Fn)load_roller_screw_2(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,nut_temp,screw_temp,share),F0,options);
        %%%���㿼��ʵ�ʹ�����϶�ĸ��Ӵ��㷨���غ�
        if (sum(sum(modify_nut))+sum(sum(modify_screw)))>0
            %%%������ĸ��/˿�ܲ๤����϶��ֵ(mm)��max_nut/max_screw
            for kk=1:number_roller %���ǹ���˿�ܵĹ���������number_roller
                for ii=2:number %������������������number
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
            %%%������ĸ��/˿�ܲ���������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_nut/modify_screw
            modify_nut=min(modify_nut,max_nut);
            modify_screw=min(modify_screw,max_screw);
            %%%��⿼�ǹ�����϶ʱ���ǹ���˿�ܸ��Ӵ�������ط����غ�
            [Fn,fval]=fsolve(@(Fn)load_roller_screw_2(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share),F0,options);
            if (sum(sum(abs(fval(2:number,1:number_roller)))>0.01)>0)||(sum(sum(abs(fval(2+number:2*number,1:number_roller)))>0.01)>0)
                h1=msgbox('���ǹ���˿�ܷ����غɼ�����ڲ�����');
                javaFrame=get(h1,'JavaFrame');
                javaFrame.setFigureIcon(javax.swing.ImageIcon('�ɻ�.jpg'));
            end
        else
            %%%����ʵ�ʹ�����϶�ĸ��Ӵ��㷨���غ�
            Fn=Fn_refer;
        end
        %%%�������ǹ���˿�ܸ��Ӵ�������ط����غ�
        Fn(:,:)=max(0,Fn(:,:));
        %%%��¼��ĸ/˿�ܵĽӴ��㷨���غɵ�����
        %%%��¼��ĸ�ĽӴ��㷨���غɵ�������Fn_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
        Fn_nut(1:number,1)=(1:number)';
        Fn_nut(1:number,2:1+number_roller)=Fn(1:number,1:number_roller);
        %%%��¼��ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
        distribution_nut(1:number,1)=(1:number)';
        distribution_nut(1:number,2:1+number_roller)=Fn_nut(1:number,2:1+number_roller)/(force_axial/number_load/ratio_nut);
        %%%˿�ܵĽӴ��㷨���غɵ�������Fn_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
        Fn_screw(1:number,1)=(1:number)';
        Fn_screw(1:number,2:1+number_roller)=Fn(1+number:2*number,1:number_roller);
        %%%��¼��ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
        distribution_screw(1:number,1)=(1:number)';
        distribution_screw(1:number,2:1+number_roller)=Fn_screw(1:number,2:1+number_roller)/(force_axial/number_load/ratio_screw);
    end
    %%%
    %%%���������ϵĽӴ�������(mm)��half_width_a_outer
    %�������ۺ�ģ��Ӱ��ϵ����E0_outer
    %˿�ܷ����غ�(N)��force_screw
    %����������֮��(1/mm)��sum_p_outer
    half_width_a_outer(:,1)=(1:number)';
    half_width_a_outer(:,2:1+number_roller)=C_a_outer*E0_outer*(Fn_screw(:,2:1+number_roller)/sum_p_outer).^(1/3);
    %%%���������ϵĽӴ��̰���(mm)��half_width_b_outer
    half_width_b_outer(:,1)=(1:number)';
    half_width_b_outer(:,2:1+number_roller)=C_b_outer*E0_outer*(Fn_screw(:,2:1+number_roller)/sum_p_outer).^(1/3);
    %%%���������ϽӴ�Ӧ��(Mpa)��stress_outer
    stress_outer(:,1)=(1:number)';
    stress_outer(:,2:1+number_roller)=1.5*Fn_screw(:,2:1+number_roller)./(pi*half_width_a_outer(:,2:1+number_roller).*half_width_b_outer(:,2:1+number_roller));
    %%%���������ϽӴ�������(mm)��delta_outer
    delta_outer(:,1)=(1:number)';
    delta_outer(:,2:1+number_roller)=C_delta_outer*(E0_outer^2)*(sum_p_outer*(Fn_screw(:,2:1+number_roller).^2)).^(1/3);
    %%%
    %%%���������ϵĽӴ�������(mm)��half_width_a_inner
    %�������ۺ�ģ��Ӱ��ϵ����E0_inner
    %˿�ܷ����غ�(N)��force_screw
    %����������֮��(1/mm)��sum_p_inner
    half_width_a_inner(:,1)=(1:number)';
    half_width_a_inner(:,2:1+number_roller)=C_a_inner*E0_inner*(Fn_nut(:,2:1+number_roller)/sum_p_inner).^(1/3);
    %%%���������ϵĽӴ��̰���(mm)��half_width_b_inner
    half_width_b_inner(:,1)=(1:number)';
    half_width_b_inner(:,2:1+number_roller)=C_b_inner*E0_inner*(Fn_nut(:,2:1+number_roller)/sum_p_inner).^(1/3);
    %%%���������ϽӴ�Ӧ��(Mpa)��stress_inner
    stress_inner(:,1)=(1:number)';
    stress_inner(:,2:1+number_roller)=1.5*Fn_nut(:,2:1+number_roller)./(pi*half_width_a_inner(:,2:1+number_roller).*half_width_b_inner(:,2:1+number_roller));
    %%%���������ϽӴ�������(mm)��delta_inner
    delta_inner(:,1)=(1:number)';
    delta_inner(:,2:1+number_roller)=C_delta_inner*(E0_inner^2)*(sum_p_inner*(Fn_nut(:,2:1+number_roller).^2)).^(1/3);
    %%%
    %%%������ĸ�غɷֲ�����ֵ��S_load_nut
    S_load_nut=(sum((Fn_nut(:,2:1+number_roller)-sum(Fn_nut(:,2:1+number_roller))/number).^2)/number).^(0.5);
    %%%����˿���غɷֲ�����ֵ��S_load_screw
    S_load_screw=(sum((Fn_screw(:,2:1+number_roller)-sum(Fn_screw(:,2:1+number_roller))/number).^2)/number).^(0.5);
else
    %%%������ĸ�ĽӴ��㷨���غɵ�������Fn_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2 �Ӵ��㷨���غ�(N) 3 ������ϵ��]
    Fn_nut=zeros(1,2);
    %%%����˿�ܵĽӴ��㷨���غɵ�������Fn_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2 �Ӵ��㷨���غ�(N) 3 ������ϵ��]
    Fn_screw=zeros(1,2);
    %%%������ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
    distribution_nut=zeros(1,2);
    %%%������ĸ�ĽӴ��㷨���غɲ�����ϵ����������distribution_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غɵĲ�����ϵ��]
    distribution_screw=zeros(1,2);
    %%%���������ϵĽӴ�������(mm)��half_width_a_outer
    half_width_a_outer=zeros(1,1);
    %%%���������ϵĽӴ��̰���(mm)��half_width_b_outer
    half_width_b_outer=zeros(1,1);
    %%%���������ϽӴ�Ӧ��(Mpa)��stress_outer
    stress_outer=zeros(1,1);
    %%%���������ϽӴ�������(mm)��delta_outer
    delta_outer=zeros(1,1);
    %%%���������ϵĽӴ�������(mm)��half_width_a_inner
    half_width_a_inner=zeros(1,1);
    %%%���������ϵĽӴ��̰���(mm)��half_width_b_inner
    half_width_b_inner=zeros(1,1);
    %%%���������ϽӴ�Ӧ��(Mpa)��stress_inner
    stress_inner=zeros(1,1);
    %%%���������ϽӴ�������(mm)��delta_inner
    delta_inner=zeros(1,1);
    %%%������ĸ�غɷֲ�����ֵ��S_load_nut
    S_load_nut=zeros(1,1);
    %%%����˿���غɷֲ�����ֵ��S_load_screw
    S_load_screw=zeros(1,1);
end
%%%
%%%�������ǹ���˿���ܵ�����
[potential_total]=potential_roller_screw(order_roller_screw,roller_screw_initial,type_load,Fn_nut,Fn_screw,ratio_nut,ratio_screw,flexible_inner,flexible_outer,B_inner,B_outer,half_width_a_inner,half_width_b_inner,stress_inner,half_width_a_outer,half_width_b_outer,stress_outer);
%���ǹ���˿���ܵ����ܣ�potential_total
end
%%%���������


%%%����ͬ���������ǹ���˿�ܵķ�����(��ĸ��ѹ��˿������)
function FF=load_roller_screw_1(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share)
%%%�������
%�������Ҳೣ���FF
%%%��������
%���ǹ���˿�ܵķ����غɣ�ע����1������Z�� ��ĸ�����غ�(N)����Z+1������2*Z�� ˿�ܷ����غ�(N)
%���ǹ���˿���ܵ������غ�(N)��force_axial
%�ݾ�(mm)��pitch_screw
%���ǹ���˿�ܵĹ���������number_roller
%��ĸ�������������ratio_nut
%˿���������������ratio_screw
%��ĸ����ģ��(Mpa)��E_n
%˿�ܵ���ģ��(Mpa)��E_s
%��������ģ��(Mpa)��E_r
%��ĸ�����(mm^2)��area_n
%˿�ܽ����(mm^2)��area_s
%���������(mm^2)��area_r
%�����ϵĽӴ�����Ӱ��ϵ����C_delta_inner
%�����ϵĽӴ�����Ӱ��ϵ����C_delta_outer
%�������ۺ�ģ��Ӱ��ϵ����E0_inner
%�������ۺ�ģ��Ӱ��ϵ����E0_outer
%����������֮��(1/mm)��sum_p_inner
%����������֮��(1/mm)��sum_p_outer
%�����Ƶ��������(mm/N)��flexible_inner
%�����Ƶ��������(mm/N)��flexible_outer
%��ĸ����������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_nut
%˿�ܲ���������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_screw
%�����غɷ���ϵ����share
%%%����ʼ
%%%���㷨���غɸ�����number
number=length(Fn(:,1))/2;
%%%���巽�����Ҳೣ���FF
FF=zeros(2*number,number_roller);
%%%������ĸ�������λ��Э����������ĸ�����غɷ�����
for kk=1:number_roller %���ǹ���˿�ܵĹ���������number_roller
    for ii=1:number %�����غɸ�����number
        if ii==1
            FF(ii,kk)=sum(max(0,Fn(1:number,kk)))-share(kk,1)*force_axial/ratio_nut;
        else
            FF(ii,kk)=10^3*(sum(sum(max(0,Fn(ii:number,1:number_roller))))*ratio_nut*pitch_screw/(area_n*E_n)-C_delta_inner*E0_inner^2*sum_p_inner^(1/3)*(max(0,Fn(ii-1,kk))^(2/3)-max(0,Fn(ii,kk))^(2/3))/ratio_nut-flexible_inner*(max(0,Fn(ii-1,kk))-max(0,Fn(ii,kk)))-(modify_nut(ii-1,kk)-modify_nut(ii,kk))+(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
    %%%
    %%%����˿���������λ��Э��������˿�ܷ����غɷ�����
    for ii=1:number %�����غɸ�����number
        if ii==1
            FF(ii+number,kk)=sum(max(0,Fn(1+number:2*number,kk)))-share(kk,1)*force_axial/ratio_screw;
        else
            FF(ii+number,kk)=10^3*(sum(sum(max(0,Fn(ii+number:2*number,1:number_roller))))*ratio_screw*pitch_screw/(area_s*E_s)-C_delta_outer*E0_outer^2*sum_p_outer^(1/3)*(max(0,Fn(ii-1+number,kk))^(2/3)-max(0,Fn(ii+number,kk))^(2/3))/ratio_screw-flexible_outer*(max(0,Fn(ii-1+number,kk))-max(0,Fn(ii+number,kk)))-(modify_screw(ii-1,kk)-modify_screw(ii,kk))-(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
end
end
%%%�������

%%%��������������ǹ���˿�ܵķ�����(��ĸ��ѹ��˿����ѹ)
function FF=load_roller_screw_2(Fn,force_axial,pitch_screw,number_roller,ratio_nut,ratio_screw,E_n,E_s,E_r,area_n,area_s,area_r,C_delta_inner,C_delta_outer,E0_inner,E0_outer,sum_p_inner,sum_p_outer,flexible_inner,flexible_outer,modify_nut,modify_screw,share)
%%%�������
%�������Ҳೣ���FF
%%%��������
%���ǹ���˿�ܵķ����غɣ�ע����1������Z�� ��ĸ�����غ�(N)����Z+1������2*Z�� ˿�ܷ����غ�(N)
%���ǹ���˿���ܵ������غ�(N)��force_axial
%�ݾ�(mm)��pitch_screw
%���ǹ���˿�ܵĹ���������number_roller
%��ĸ�������������ratio_nut
%˿���������������ratio_screw
%��ĸ����ģ��(Mpa)��E_n
%˿�ܵ���ģ��(Mpa)��E_s
%��������ģ��(Mpa)��E_r
%��ĸ�����(mm^2)��area_n
%˿�ܽ����(mm^2)��area_s
%���������(mm^2)��area_r
%�����ϵĽӴ�����Ӱ��ϵ����C_delta_inner
%�����ϵĽӴ�����Ӱ��ϵ����C_delta_outer
%�������ۺ�ģ��Ӱ��ϵ����E0_inner
%�������ۺ�ģ��Ӱ��ϵ����E0_outer
%����������֮��(1/mm)��sum_p_inner
%����������֮��(1/mm)��sum_p_outer
%�����Ƶ��������(mm/N)��flexible_inner
%�����Ƶ��������(mm/N)��flexible_outer
%��ĸ����������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_nut
%˿�ܲ���������(ע����ĸ���ض�Ϊ��ſ�ʼ�ˣ��ҹ������϶)(mm)��modify_screw
%�����غɷ���ϵ����share
%%%����ʼ
%%%���㷨���غɸ�����number
number=length(Fn(:,1))/2;
%%%���巽�����Ҳೣ���FF
FF=zeros(2*number,number_roller);
%%%������ĸ�������λ��Э����������ĸ�����غɷ�����
for kk=1:number_roller %���ǹ���˿�ܵĹ���������number_roller
    for ii=1:number %�����غɸ�����number
        if ii==1
            FF(ii,kk)=sum(max(0,Fn(1:number,kk)))-share(kk,1)*force_axial/ratio_nut;
        else
            FF(ii,kk)=10^3*(sum(sum(max(0,Fn(ii:number,1:number_roller))))*ratio_nut*pitch_screw/(area_n*E_n)-C_delta_inner*E0_inner^2*sum_p_inner^(1/3)*(max(0,Fn(ii-1,kk))^(2/3)-max(0,Fn(ii,kk))^(2/3))/ratio_nut-flexible_inner*(max(0,Fn(ii-1,kk))-max(0,Fn(ii,kk)))-(modify_nut(ii-1,kk)-modify_nut(ii,kk))+(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
    %%%
    %%%����˿���������λ��Э��������˿�ܷ����غɷ�����
    for ii=1:number %�����غɸ�����number
        if ii==1
            FF(ii+number,kk)=sum(max(0,Fn(1+number:2*number,kk)))-share(kk,1)*force_axial/ratio_screw;
        else
            FF(ii+number,kk)=10^3*(sum(sum(max(0,Fn(1+number:ii-1+number,1:number_roller))))*ratio_screw*pitch_screw/(area_s*E_s)-C_delta_outer*E0_outer^2*sum_p_outer^(1/3)*(max(0,Fn(ii+number,kk))^(2/3)-max(0,Fn(ii-1+number,kk))^(2/3))/ratio_screw-flexible_outer*(max(0,Fn(ii+number,kk))-max(0,Fn(ii-1+number,kk)))-(modify_screw(ii,kk)-modify_screw(ii-1,kk))+(sum(max(0,Fn(ii:number,kk)))*ratio_nut-sum(max(0,Fn(ii+number:2*number,kk)))*ratio_screw)*pitch_screw/(area_r*E_r));
        end
    end
end
end
%%%�������

%%%�������ǹ���˿��������/�����Ƶ��������
function [flexible_inner,flexible_outer]=flexible_roller_screw(roller_screw_temp)
%%%�������
%�����Ƶ��������(mm/N)��flexible_inner
%�����Ƶ��������(mm/N)��flexible_outer
%%%��������
%���ǹ���˿�ܻ���������roller_screw_temp��[1 ���ǹ���˿����� 2 ���ǹ���˿�ܵ�����(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw 3 ��������,number_roller 4 ˿������ͷ��,N_thread_s 5 ��ĸ����ͷ��,N_thread_n 6 ��������ͷ��,N_thread_r
%7 ˿����������(-1��ʾ������1��ʾ����),direct_screw 8 ��ĸ��������(-1��ʾ������1��ʾ����),direct_nut 9 ������������(-1��ʾ������1��ʾ����),direct_roller 10 ˿����������(rad)��rise_angle_s 11 ��ĸ��������(rad)��rise_angle_n 12 ������������(rad)��rise_angle_r
%13 ��1��������X��н�(rad)��angle_roller 14 �����Ƶ���Ħ��ϵ����f_inner 15 �����Ƶ���Ħ��ϵ����f_outer 16 �ݾ�(mm)��pitch_screw 17 ���ͽ�(rad)��angle_screw 18 ��������Բ���뾶(mm)��radius_roller 19 ��������Բ�Ľ�������ϵX��ֵ(mm)��X_center_r 20 ��������Բ�Ľ�������ϵY��ֵ(mm)��Y_center_r
%21 ˿����������ϵ����cutter_top_s 22 ˿����������ϵ����cutter_bottom_s 23 ��ĸ��������ϵ����cutter_top_n 24 ��ĸ��������ϵ����cutter_bottom_n 25 ������������ϵ����cutter_top_r 26 ������������ϵ����cutter_bottom_r 27 ˿�����ư���������(mm)��reduce_s 28 ��ĸ���ư���������(mm)��reduce_n 29 �������ư���������(mm)��reduce_r 30 ��
%31 ˿�ܵ����Ƴݶ���(mm)��addendum_s 32 ˿�ܵ����Ƴݸ���(mm)��dedendum_s 33 ��ĸ�����Ƴݶ���(mm)��addendum_n 34 ��ĸ�����Ƴݸ���(mm)��dedendum_n 35 ���������Ƴݶ���(mm)��addendum_r 36 ���������Ƴݸ���(mm)��dedendum_r
%37 ˿��ʵ�ʶ���(mm)��D_top_s 38 ˿��ʵ�ʸ���(mm)��D_bottom_s 39 ��ĸʵ�ʶ���(mm)��D_top_n
%40 ��ĸʵ�ʸ���(mm)��D_bottom_n 41 ����ʵ�ʶ���(mm)��D_top_r 42 ����ʵ�ʸ���(mm)��D_bottom_r
%43 ˿���о�(mm),D_pitch_s ,44 ��ĸ�о�(mm),D_pitch_n, 45 �����о�(mm)��D_pitch_r
%46 ˿���᳤��(mm)��length_shaft_s 47 ��ĸ�᳤��(mm)��length_shaft_n 48 �����᳤��(mm)��length_shaft_r 49 ˿�����⾶(mm)��diameter_outer_s 50 ��ĸ���⾶(mm)��diameter_outer_n 51 �������⾶(mm)��diameter_outer_r 52 ˿�����ھ�(mm)��diameter_inner_s 53 ��ĸ���ھ�(mm)��diameter_inner_n 54 �������ھ�(mm)��diameter_inner_r
%55 ˿�����Ƴ���(mm)��length_thread_s 56 ��ĸ���Ƴ���(mm)��length_thread_n 57 �������Ƴ���(mm)��length_thread_r 58 �����ϵ���Ч���Ƴ���(mm)��length_mesh_inner 59 �����ϵ���Ч���Ƴ���(mm)��length_mesh_outer 60 ��
%61 ����ֿ��(mm)��width_gear_left 62 �ҳ��ֿ��(mm)��width_gear_left 63 �󱣳ּܿ��(mm)��width_carrier_left 64 �󱣳ּܿ��(mm)��width_carrier_right
%65 ��ĸ����������˿������˵�λ��(mm)��delta_n_s 66 ��������������˿������˵�λ��(mm)��delta_r_s 67 ˿��������������˿������˵�λ��(mm),delta_thread_s 68 ��ĸ��������������ĸ����˵�λ��(mm),delta_thread_n 69 ���������������ڹ�������˵�λ��(mm),delta_thread_r 70 ����Ԥ���غ�(N)��preload_axial
%71 ˿������ܶ�(t/mm^-9)��density_s 72 ��ĸ����ܶ�(t/mm^-9)��density_n 73 ��������ܶ�(t/mm^-9)��density_r 74 ˿�ܵ���ģ��(Mpa)��E_s 75 ��ĸ����ģ��(Mpa)��E_n 76 ��������ģ��(Mpa)��E_r 77 ˿�ܲ��ɱȣ�possion_s 78 ��ĸ���ɱȣ�possion_n 79 �������ɱȣ�possion_r
%80 ������������ڹ��������λ��(mm)��delta_gear_left 81 �ҳ����������ڹ��������λ��(mm)��delta_gear_right 82 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_left 83 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_right
%84 �󱣳ּ��⾶(mm)��outer_carrier_left 85 �󱣳ּ��ھ�(mm)��inner_carrier_left 86 �ұ��ּ��⾶(mm)��outer_carrier_right 87 �ұ��ּ��ھ�(mm)��inner_carrier_right 88 ���ǹ���˿�Ƿ�ָ��Ч��(0��ʾ��ָ����1��ʾָ��),sign_efficiency 89 ��������Ч��(��Ԥ���غ�)��advance_efficiency 90 ��������Ч��(��Ԥ���غ�)��reverse_efficiency
%91 ���ַ���ģ��(mm)��m_n 92 �ڳ�Ȧ������z_r 93 ̫���ֳ�����z_s 94 �����ֳ�����z_p 95 ѹ����(rad)��press_angle 96 ����ѹ����(rad)��tran_press_angle 97 ������(rad)��helix_angle 98 �ڳ�Ȧ�����λϵ����n_r 99 ̫���ַ����λϵ����n_s 100 �����ַ����λϵ����n_p
%101 �������ľ�(mm)��work_center 102 ������ϵ�����Ͻǣ�mesh_angle(rad) 103 ������ϵ�����Ͻǣ�mesh_angle(rad) 104 �ڳ�Ȧ�ݶ�Բ��tran_ad_dia_r��mm�� 105 ̫���ֳݶ�Բ��tran_ad_dia_s��mm�� 106 �����ֳݶ�Բ��tran_ad_dia_p��mm�� 107 �ڳ�Ȧ�ݸ�Բ��tran_de_dia_r��mm�� 108 ̫���ֳݸ�Բ��tran_de_dia_s��mm�� 109 �����ֳݸ�Բ��tran_de_dia_p��mm��]
%110 �ڳ�Ȧ�Ľ�Բ��mm����tran_pitch_dia_r 111 ̫���ֵĽ�Բ��mm����tran_pitch_dia_s 112 �����ֵĽ�Բ��mm����tran_pitch_dia_p 113-120 ��
%121 ��ĸ������(mm)��width_web_n 122 ��ĸ�������ľ���ĸ����˾���(mm)��delta_web_n 123 ˿���������(mm)��width_left_web_s 124 ˿����������ľ�˿������˾���(mm)��delta_left_web_s 125 ˿���ҷ�����(mm)��width_right_web_s 126 ˿���ҷ������ľ�˿������˾���(mm)��delta_right_web_s 127-130 ��
%131 ��ĸ��Ե�����,shaft_self_nut 132 ˿����Ե����ţ�shaft_self_screw 133 ������Ե����ţ�shaft_self_roller,134-140 ��
%141 �����ֵ��߳ݶ���ϵ����142 �����ֵ��߶�϶ϵ����143 �����ֳݶ���������144 �ڳ�Ȧ���߳ݶ���ϵ����145 �ڳ�Ȧ���߶�϶ϵ����146 �ڳ�Ȧ�ݶ�������,147 ̫���ֵ��߳ݶ���ϵ����148 ̫���ֵ��߶�϶ϵ����149 ̫���ֳݶ�������
%150-160 �� 161 ��Ч�����������������ĸ����˵�λ��(mm)��thread_inner_nut 162 ��Ч��������������ڹ�������˵�λ��(mm)��thread_inner_roller 163 ��Ч���������������˿������˵�λ��(mm)��thread_outer_screw 164 ��Ч��������������ڹ�������˵�λ��(mm)��thread_outer_roller]
%%%����ʼ
%%%��¼�ݾ�(mm)��pitch_screw
pitch_screw=roller_screw_temp(1,16);
%%%��¼���ͽ�(rad)��angle_screw
angle_screw=roller_screw_temp(1,17);
%%%��¼˿����������ϵ����cutter_bottom_s
cutter_bottom_s=roller_screw_temp(1,22);
%%%��¼��ĸ��������ϵ����cutter_bottom_n
cutter_bottom_n=roller_screw_temp(1,24);
%%%��¼������������ϵ����cutter_bottom_r
cutter_bottom_r=roller_screw_temp(1,26);
%%%��¼˿�����ư���������(mm)��reduce_s
reduce_s=roller_screw_temp(1,27);
%%%��¼��ĸ���ư���������(mm)��reduce_n
reduce_n=roller_screw_temp(1,28);
%%%��¼�������ư���������(mm)��reduce_r
reduce_r=roller_screw_temp(1,29);
%%%��¼˿���о�(mm)��D_pitch_s
D_pitch_s=roller_screw_temp(1,43);
%%%��¼��ĸ�о�(mm),D_pitch_n
D_pitch_n=roller_screw_temp(1,44);
%%%��¼�����о�(mm)��D_pitch_r
D_pitch_r=roller_screw_temp(1,45);
%%%��ĸ���⾶(mm)��diameter_outer_n
diameter_outer_n=roller_screw_temp(1,50);
%%%��¼˿�ܵ���ģ��(Mpa)��E_s
E_s=roller_screw_temp(1,74);
%%%��¼��ĸ����ģ��(Mpa)��E_n
E_n=roller_screw_temp(1,75);
%%%��¼��������ģ��(Mpa)��E_r
E_r=roller_screw_temp(1,76);
%%%��¼˿�ܲ��ɱȣ�possion_s
possion_s=roller_screw_temp(1,77);
%%%��¼��ĸ���ɱȣ�possion_n
possion_n=roller_screw_temp(1,78);
%%%��¼�������ɱȣ�possion_r
possion_r=roller_screw_temp(1,79);
%%%
%%%�������Ƹ߶ȣ�high_thread
high_thread=(0.5*pitch_screw)/tan(0.5*angle_screw);
%%%������ĸ�о�����(mm)��b_nut
b_nut=0.5*pitch_screw-2*reduce_n;
%%%������ĸ�׾�����(mm)��a_nut
a_nut=pitch_screw*(1-cutter_bottom_n)-2*reduce_n;
%%%������ĸ������(mm)��c_nut
c_nut=(0.5-cutter_bottom_n)*high_thread;
%%%��������о�����(mm)��b_roller
b_roller=0.5*pitch_screw-2*reduce_r;
%%%��������׾�����(mm)��a_roller
a_roller=pitch_screw*(1-cutter_bottom_r)-2*reduce_r;
%%%�������������(mm)��c_roller
c_roller=(0.5-cutter_bottom_r)*high_thread;
%%%����˿���о�����(mm)��b_screw
b_screw=0.5*pitch_screw-2*reduce_s;
%%%����˿�ܵ׾�����(mm)��a_screw
a_screw=pitch_screw*(1-cutter_bottom_s)-2*reduce_s;
%%%����˿��������(mm)��c_screw
c_screw=(0.5-cutter_bottom_s)*high_thread;
%%%
%%%������ĸ�������(mm/N)��flexible_nut
flexible_nut_1=(1-possion_n^2)*(3/(4*E_n))*((1-(2-b_nut/a_nut)^2+2*log(a_nut/b_nut))*cot(0.5*angle_screw)^3-4*(c_nut/a_nut)^2*tan(0.5*angle_screw));
flexible_nut_2=(1+possion_n)*(6/(5*E_n))*cot(0.5*angle_screw)*log(a_nut/b_nut);
flexible_nut_3=(1-possion_n^2)*(12*c_nut/(pi*E_n*a_nut^2))*(c_nut-0.5*b_nut*tan(0.5*angle_screw));
flexible_nut_4=(1-possion_n^2)*(2/(pi*E_n))*((pitch_screw/a_nut)*log((pitch_screw+0.5*a_nut)/(pitch_screw-0.5*a_nut))+0.5*log(4*pitch_screw^2/a_nut^2-1));
flexible_nut_5=((diameter_outer_n^2+D_pitch_n^2)/(diameter_outer_n^2-D_pitch_n^2)+possion_n)*0.5*tan(0.5*angle_screw)^2*(D_pitch_n/pitch_screw)/E_n;
flexible_nut=flexible_nut_1+flexible_nut_2+flexible_nut_3+flexible_nut_4+flexible_nut_5;
%%%
%%%����˿���������(mm/N)��flexible_screw
flexible_screw_1=(1-possion_s^2)*(3/(4*E_s))*((1-(2-b_screw/a_screw)^2+2*log(a_screw/b_screw))*cot(0.5*angle_screw)^3-4*(c_screw/a_screw)^2*tan(0.5*angle_screw));
flexible_screw_2=(1+possion_s)*(6/(5*E_s))*cot(0.5*angle_screw)*log(a_screw/b_screw);
flexible_screw_3=(1-possion_s^2)*(12*c_screw/(pi*E_s*a_screw^2))*(c_screw-0.5*b_screw*tan(0.5*angle_screw));
flexible_screw_4=(1-possion_s^2)*(2/(pi*E_s))*((pitch_screw/a_screw)*log((pitch_screw+0.5*a_screw)/(pitch_screw-0.5*a_screw))+0.5*log(4*pitch_screw^2/a_screw^2-1));
flexible_screw_5=(1-possion_s)*0.5*tan(0.5*angle_screw)^2*(D_pitch_s/pitch_screw)/E_s;
flexible_screw=flexible_screw_1+flexible_screw_2+flexible_screw_3+flexible_screw_4+flexible_screw_5;
%%%
%%%��������������(mm/N)��flexible_roller
flexible_roller_1=(1-possion_r^2)*(3/(4*E_r))*((1-(2-b_roller/a_roller)^2+2*log(a_roller/b_roller))*cot(0.5*angle_screw)^3-4*(c_roller/a_roller)^2*tan(0.5*angle_screw));
flexible_roller_2=(1+possion_r)*(6/(5*E_r))*cot(0.5*angle_screw)*log(a_roller/b_roller);
flexible_roller_3=(1-possion_r^2)*(12*c_roller/(pi*E_r*a_roller^2))*(c_roller-0.5*b_roller*tan(0.5*angle_screw));
flexible_roller_4=(1-possion_r^2)*(2/(pi*E_r))*((pitch_screw/a_roller)*log((pitch_screw+0.5*a_roller)/(pitch_screw-0.5*a_roller))+0.5*log(4*pitch_screw^2/a_roller^2-1));
flexible_roller_5=(1-possion_r)*0.5*tan(0.5*angle_screw)^2*(D_pitch_r/pitch_screw)/E_r;
flexible_roller=flexible_roller_1+flexible_roller_2+flexible_roller_3+flexible_roller_4+flexible_roller_5;
%%%
%%%���������Ƶ��������(mm/N)��flexible_inner
flexible_inner=abs(flexible_nut+flexible_roller)/1;
%%%
%%%���������Ƶ��������(mm/N)��flexible_outer
flexible_outer=abs(flexible_screw+flexible_roller)/1;
end
%%%�������

%%%�������ǹ���˿�ܵ�����
function [potential_total,potential_axial,potential_bend,potential_compress]=potential_roller_screw(order_roller_screw,roller_screw_initial,type_load,Fn_nut,Fn_screw,ratio_nut,ratio_screw,flexible_inner,flexible_outer,B_inner,B_outer,half_width_a_inner,half_width_b_inner,stress_inner,half_width_a_outer,half_width_b_outer,stress_outer)
%%%�������
%���ǹ���˿���ܵ����ܣ�potential_total
%��ĸ/˿��/������������/ѹ�������ܺͣ�potential_axial
%�������������ܺͣ�potential_bend
%����ѹ�������ܺͣ�potential_compress
%%%��������
%���ǹ���˿�ܵ���ţ�order_roller_screw
%���ǹ���˿�ܻ���������roller_screw_initial��[1 ���ǹ���˿����� 2 ���ǹ���˿�ܵ�����(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw 3 ��������,number_roller 4 ˿������ͷ��,N_thread_s 5 ��ĸ����ͷ��,N_thread_n 6 ��������ͷ��,N_thread_r
%7 ˿����������(-1��ʾ������1��ʾ����),direct_screw 8 ��ĸ��������(-1��ʾ������1��ʾ����),direct_nut 9 ������������(-1��ʾ������1��ʾ����),direct_roller 10 ˿����������(rad)��rise_angle_s 11 ��ĸ��������(rad)��rise_angle_n 12 ������������(rad)��rise_angle_r
%13 ��1��������X��н�(rad)��angle_roller 14 �����Ƶ���Ħ��ϵ����f_inner 15 �����Ƶ���Ħ��ϵ����f_outer 16 �ݾ�(mm)��pitch_screw 17 ���ͽ�(rad)��angle_screw 18 ��������Բ���뾶(mm)��radius_roller 19 ��������Բ�Ľ�������ϵX��ֵ(mm)��X_center_r 20 ��������Բ�Ľ�������ϵY��ֵ(mm)��Y_center_r
%21 ˿����������ϵ����cutter_top_s 22 ˿����������ϵ����cutter_bottom_s 23 ��ĸ��������ϵ����cutter_top_n 24 ��ĸ��������ϵ����cutter_bottom_n 25 ������������ϵ����cutter_top_r 26 ������������ϵ����cutter_bottom_r 27 ˿�����ư���������(mm)��reduce_s 28 ��ĸ���ư���������(mm)��reduce_n 29 �������ư���������(mm)��reduce_r 30 ��
%31 ˿�ܵ����Ƴݶ���(mm)��addendum_s 32 ˿�ܵ����Ƴݸ���(mm)��dedendum_s 33 ��ĸ�����Ƴݶ���(mm)��addendum_n 34 ��ĸ�����Ƴݸ���(mm)��dedendum_n 35 ���������Ƴݶ���(mm)��addendum_r 36 ���������Ƴݸ���(mm)��dedendum_r
%37 ˿��ʵ�ʶ���(mm)��D_top_s 38 ˿��ʵ�ʸ���(mm)��D_bottom_s 39 ��ĸʵ�ʶ���(mm)��D_top_n
%40 ��ĸʵ�ʸ���(mm)��D_bottom_n 41 ����ʵ�ʶ���(mm)��D_top_r 42 ����ʵ�ʸ���(mm)��D_bottom_r
%43 ˿���о�(mm),D_pitch_s ,44 ��ĸ�о�(mm),D_pitch_n, 45 �����о�(mm)��D_pitch_r
%46 ˿���᳤��(mm)��length_shaft_s 47 ��ĸ�᳤��(mm)��length_shaft_n 48 �����᳤��(mm)��length_shaft_r 49 ˿�����⾶(mm)��diameter_outer_s 50 ��ĸ���⾶(mm)��diameter_outer_n 51 �������⾶(mm)��diameter_outer_r 52 ˿�����ھ�(mm)��diameter_inner_s 53 ��ĸ���ھ�(mm)��diameter_inner_n 54 �������ھ�(mm)��diameter_inner_r
%55 ˿�����Ƴ���(mm)��length_thread_s 56 ��ĸ���Ƴ���(mm)��length_thread_n 57 �������Ƴ���(mm)��length_thread_r 58 �����ϵ���Ч���Ƴ���(mm)��length_mesh_inner 59 �����ϵ���Ч���Ƴ���(mm)��length_mesh_outer 60 ��
%61 ����ֿ��(mm)��width_gear_left 62 �ҳ��ֿ��(mm)��width_gear_left 63 �󱣳ּܿ��(mm)��width_carrier_left 64 �󱣳ּܿ��(mm)��width_carrier_right
%65 ��ĸ����������˿������˵�λ��(mm)��delta_n_s 66 ��������������˿������˵�λ��(mm)��delta_r_s 67 ˿��������������˿������˵�λ��(mm),delta_thread_s 68 ��ĸ��������������ĸ����˵�λ��(mm),delta_thread_n 69 ���������������ڹ�������˵�λ��(mm),delta_thread_r 70 ����Ԥ���غ�(N)��preload_axial
%71 ˿������ܶ�(t/mm^-9)��density_s 72 ��ĸ����ܶ�(t/mm^-9)��density_n 73 ��������ܶ�(t/mm^-9)��density_r 74 ˿�ܵ���ģ��(Mpa)��E_s 75 ��ĸ����ģ��(Mpa)��E_n 76 ��������ģ��(Mpa)��E_r 77 ˿�ܲ��ɱȣ�possion_s 78 ��ĸ���ɱȣ�possion_n 79 �������ɱȣ�possion_r
%80 ������������ڹ��������λ��(mm)��delta_gear_left 81 �ҳ����������ڹ��������λ��(mm)��delta_gear_right 82 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_left 83 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_right
%84 �󱣳ּ��⾶(mm)��outer_carrier_left 85 �󱣳ּ��ھ�(mm)��inner_carrier_left 86 �ұ��ּ��⾶(mm)��outer_carrier_right 87 �ұ��ּ��ھ�(mm)��inner_carrier_right 88 ���ǹ���˿�Ƿ�ָ��Ч��(0��ʾ��ָ����1��ʾָ��),sign_efficiency 89 ��������Ч��(��Ԥ���غ�)��advance_efficiency 90 ��������Ч��(��Ԥ���غ�)��reverse_efficiency
%91 ���ַ���ģ��(mm)��m_n 92 �ڳ�Ȧ������z_r 93 ̫���ֳ�����z_s 94 �����ֳ�����z_p 95 ѹ����(rad)��press_angle 96 ����ѹ����(rad)��tran_press_angle 97 ������(rad)��helix_angle 98 �ڳ�Ȧ�����λϵ����n_r 99 ̫���ַ����λϵ����n_s 100 �����ַ����λϵ����n_p
%101 �������ľ�(mm)��work_center 102 ������ϵ�����Ͻǣ�mesh_angle(rad) 103 ������ϵ�����Ͻǣ�mesh_angle(rad) 104 �ڳ�Ȧ�ݶ�Բ��tran_ad_dia_r��mm�� 105 ̫���ֳݶ�Բ��tran_ad_dia_s��mm�� 106 �����ֳݶ�Բ��tran_ad_dia_p��mm�� 107 �ڳ�Ȧ�ݸ�Բ��tran_de_dia_r��mm�� 108 ̫���ֳݸ�Բ��tran_de_dia_s��mm�� 109 �����ֳݸ�Բ��tran_de_dia_p��mm��]
%110 �ڳ�Ȧ�Ľ�Բ��mm����tran_pitch_dia_r 111 ̫���ֵĽ�Բ��mm����tran_pitch_dia_s 112 �����ֵĽ�Բ��mm����tran_pitch_dia_p 113-120 ��
%121 ��ĸ������(mm)��width_web_n 122 ��ĸ�������ľ���ĸ����˾���(mm)��delta_web_n 123 ˿���������(mm)��width_left_web_s 124 ˿����������ľ�˿������˾���(mm)��delta_left_web_s 125 ˿���ҷ�����(mm)��width_right_web_s 126 ˿���ҷ������ľ�˿������˾���(mm)��delta_right_web_s 127-130 ��
%131 ��ĸ��Ե�����,shaft_self_nut 132 ˿����Ե����ţ�shaft_self_screw 133 ������Ե����ţ�shaft_self_roller,134-140 ��
%141 �����ֵ��߳ݶ���ϵ����142 �����ֵ��߶�϶ϵ����143 �����ֳݶ���������144 �ڳ�Ȧ���߳ݶ���ϵ����145 �ڳ�Ȧ���߶�϶ϵ����146 �ڳ�Ȧ�ݶ�������,147 ̫���ֵ��߳ݶ���ϵ����148 ̫���ֵ��߶�϶ϵ����149 ̫���ֳݶ�������
%150-160 �� 161 ��Ч�����������������ĸ����˵�λ��(mm)��thread_inner_nut 162 ��Ч��������������ڹ�������˵�λ��(mm)��thread_inner_roller 163 ��Ч���������������˿������˵�λ��(mm)��thread_outer_screw 164 ��Ч��������������ڹ�������˵�λ��(mm)��thread_outer_roller]
%���ǹ���˿�����ط�ʽ��type_load��1��ʾͬ�����أ�2��ʾ�������
%��ĸ�ĽӴ��㷨���غɵ�������Fn_nut��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
%˿�ܵĽӴ��㷨���غɵ�������Fn_screw��[1 �Ӵ������(ע:��Ŵ���ĸ���ض˿�ʼ) 2��1+�������� �Ӵ��㷨���غ�(N)]
%��ĸ�������������ratio_nut
%˿���������������ratio_screw
%�����Ƶ��������(mm/N)��flexible_inner
%�����Ƶ��������(mm/N)��flexible_outer
%������ѹ�����ܸ���ֵ��B_inner
%������ѹ�����ܸ���ֵ��B_outer
%�����ϵĽӴ�������(mm)��half_width_a_inner
%�����ϵĽӴ��̰���(mm)��half_width_b_inner
%�����ϽӴ�Ӧ��(Mpa)��stress_inner
%�����ϵĽӴ�������(mm)��half_width_a_outer
%�����ϵĽӴ��̰���(mm)��half_width_b_outer
%�����ϽӴ�Ӧ��(Mpa)��stress_outer
%%%����ʼ
%%%�������ǹ���˿����Ե�������ξ���
[roller_screw_segment,number]=roller_screw_segment_produce(order_roller_screw,roller_screw_initial);
%���ǹ���˿����Ե�������ξ���roller_screw_segment��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
%��Ч����������������number
%%%�������ǹ���˿�ܵ���������ѹ��Ťת��ȫ������ϵ�ĸնȾ���shaft_stiffness
[roller_screw_stiffness]=roller_screw_stiffness_produce(order_roller_screw,roller_screw_segment);
%���ǹ���˿�ܵ������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���roller_screw_stiffness��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
%%%��¼���ǹ���˿����ز���
%���ǹ���˿����ţ�order_roller_screw
id_roller_screw=(roller_screw_initial(:,1)==order_roller_screw);
roller_screw_temp=roller_screw_initial(id_roller_screw,:);
%%%��¼����������number_roller
number_roller=roller_screw_temp(1,3);
%%%
%%%������ĸ����/ѹ�����ܣ�potential_nut
%%%������ĸ�������غ�������Axial_nut
Axial_nut=zeros(6*(2+number),1);
for ii=1:number %��Ч����������������number
    Axial_nut(6*ii+3,1)=-1*sum(Fn_nut(ii,2:1+number_roller))*ratio_nut;
end
Axial_nut(3,1)=sum(sum(Fn_nut(:,2:1+number_roller)))*ratio_nut;
%%%��¼��ĸ�ĸնȾ���stiffness_nut
id_stiffness_nut=(roundn(roller_screw_stiffness(:,1),-9)==roundn(fix(order_roller_screw)+0.0001,-9));
stiffness_nut=roller_screw_stiffness(id_stiffness_nut,2:1+6*(2+number));
%%%������ĸ����Ⱦ���flexible_nut
flexible_nut=stiffness_nut^-1;
%%%������ĸ����/ѹ�����ܣ�potential_nut
potential_nut=0.5*Axial_nut'*flexible_nut*Axial_nut;
%%%
%%%����˿������/ѹ�����ܣ�potential_screw
%%%����˿�ܵ������غ�������Axial_screw
Axial_screw=zeros(6*(2+number),1);
for ii=1:number %��Ч����������������number
    Axial_screw(6*ii+3,1)=sum(Fn_screw(ii,2:1+number_roller))*ratio_screw;
end
if type_load==1 %���ǹ���˿�����ط�ʽ��type_load��1��ʾͬ�����أ�2��ʾ�������
    Axial_screw(3,1)=-1*sum(sum(Fn_screw(:,2:1+number_roller)))*ratio_screw;
elseif type_load==2 %���ǹ���˿�����ط�ʽ��type_load��1��ʾͬ�����أ�2��ʾ�������
    Axial_screw(6*(1+number)+3,1)=-1*sum(sum(Fn_screw(:,2:1+number_roller)))*ratio_screw;
end
%%%��¼˿�ܵĸնȾ���stiffness_screw
id_stiffness_screw=(roundn(roller_screw_stiffness(:,1),-9)==roundn(fix(order_roller_screw)+0.0002,-9));
stiffness_screw=roller_screw_stiffness(id_stiffness_screw,2:1+6*(2+number));
%%%����˿�ܵ���Ⱦ���flexible_screw
flexible_screw=stiffness_screw^-1;
%%%����˿������/ѹ�����ܣ�potential_screw
potential_screw=0.5*Axial_screw'*flexible_screw*Axial_screw;
%%%
%%%�����������/ѹ�����ܣ�potential_roller
potential_roller=0;
%%%��¼�����ĸնȾ���stiffness_roller
id_stiffness_roller=(roundn(roller_screw_stiffness(:,1),-9)==roundn(fix(order_roller_screw)+0.0004,-9));
stiffness_roller=roller_screw_stiffness(id_stiffness_roller,2:1+6*(2+number));
%%%�����������Ⱦ���flexible_roller
flexible_roller=stiffness_roller^-1;
%%%�����������/ѹ�����ܣ�potential_screw
for kk=1:number_roller %����������number_roller
    Axial_roller=zeros(6*(2+number),1);
    for ii=1:number %��Ч����������������number
        Axial_roller(6*ii+3,1)=Fn_nut(ii,1+kk)*ratio_nut-Fn_screw(ii,1+kk)*ratio_screw;
    end
    %%%�����������/ѹ�����ܣ�potential_roller
    potential_roller=potential_roller+0.5*Axial_roller'*flexible_roller*Axial_roller;
end
%%%
%%%������ĸ/˿��/������������/ѹ�������ܺͣ�potential_axial
potential_axial=potential_nut+potential_screw+potential_roller;
%%%
%%%���������������������ܣ�potential_bend_inner
%�����Ƶ��������(mm/N)��flexible_inner
potential_bend_inner=sum(sum(0.5*Fn_nut(1:number,2:1+number_roller).*flexible_inner.*Fn_nut(1:number,2:1+number_roller)));
%%%���������������������ܣ�potential_bend_outer
%�����Ƶ��������(mm/N)��flexible_outer
potential_bend_outer=sum(sum(0.5*Fn_screw(1:number,2:1+number_roller).*flexible_outer.*Fn_screw(1:number,2:1+number_roller)));
%%%�����������������ܺͣ�potential_bend
potential_bend=potential_bend_inner+potential_bend_outer;
%%%
%%%��������������ѹ�����ܣ�potential_compress_inner
%������ѹ�����ܸ���ֵ��B_inner
%�����ϵĽӴ�������(mm)��half_width_a_inner
%�����ϵĽӴ��̰���(mm)��half_width_b_inner
%�����ϽӴ�Ӧ��(Mpa)��stress_inner
potential_compress_inner=sum(sum(0.25*pi*B_inner*half_width_a_inner(1:number,2:1+number_roller).*(half_width_b_inner(1:number,2:1+number_roller).^2).*stress_inner(1:number,2:1+number_roller)));
%%%��������������ѹ�����ܣ�potential_compress_outer
%������ѹ�����ܸ���ֵ��B_outer
%�����ϵĽӴ�������(mm)��half_width_a_outer
%�����ϵĽӴ��̰���(mm)��half_width_b_outer
%�����ϽӴ�Ӧ��(Mpa)��stress_outer
potential_compress_outer=sum(sum(0.25*pi*B_outer*half_width_a_outer(1:number,2:1+number_roller).*(half_width_b_outer(1:number,2:1+number_roller).^2).*stress_outer(1:number,2:1+number_roller)));
%%%��������ѹ�������ܺͣ�potential_compress
potential_compress=potential_compress_inner+potential_compress_outer;
%%%
%%%�������ǹ���˿���ܵ����ܣ�potential_total
potential_total=potential_axial+potential_bend+potential_compress;
end
%%%�������

%%%�������ǹ���˿����Ե�������ξ���
function [roller_screw_segment,number]=roller_screw_segment_produce(order_roller_screw,roller_screw_initial)
%%%�������
%���ǹ���˿����Ե�������ξ���roller_screw_segment��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
%��Ч����������������number
%%%��������
%���ǹ���˿����ţ�order_roller_screw
%���ǹ���˿�ܻ���������roller_screw_initial��[1 ���ǹ���˿����� 2 ���ǹ���˿�ܵ�����(1��ʾ��׼ʽ��2��ʾ����ʽ),type_roller_screw 3 ��������,number_roller 4 ˿������ͷ��,N_thread_s 5 ��ĸ����ͷ��,N_thread_n 6 ��������ͷ��,N_thread_r
%7 ˿����������(-1��ʾ������1��ʾ����),direct_screw 8 ��ĸ��������(-1��ʾ������1��ʾ����),direct_nut 9 ������������(-1��ʾ������1��ʾ����),direct_roller 10 ˿����������(rad)��rise_angle_s 11 ��ĸ��������(rad)��rise_angle_n 12 ������������(rad)��rise_angle_r
%13 ��1��������X��н�(rad)��angle_roller 14 �����Ƶ���Ħ��ϵ����f_inner 15 �����Ƶ���Ħ��ϵ����f_outer 16 �ݾ�(mm)��pitch_screw 17 ���ͽ�(rad)��angle_screw 18 ��������Բ���뾶(mm)��radius_roller 19 ��������Բ�Ľ�������ϵX��ֵ(mm)��X_center_r 20 ��������Բ�Ľ�������ϵY��ֵ(mm)��Y_center_r
%21 ˿����������ϵ����cutter_top_s 22 ˿����������ϵ����cutter_bottom_s 23 ��ĸ��������ϵ����cutter_top_n 24 ��ĸ��������ϵ����cutter_bottom_n 25 ������������ϵ����cutter_top_r 26 ������������ϵ����cutter_bottom_r 27 ˿�����ư���������(mm)��reduce_s 28 ��ĸ���ư���������(mm)��reduce_n 29 �������ư���������(mm)��reduce_r 30 ��
%31 ˿�ܵ����Ƴݶ���(mm)��addendum_s 32 ˿�ܵ����Ƴݸ���(mm)��dedendum_s 33 ��ĸ�����Ƴݶ���(mm)��addendum_n 34 ��ĸ�����Ƴݸ���(mm)��dedendum_n 35 ���������Ƴݶ���(mm)��addendum_r 36 ���������Ƴݸ���(mm)��dedendum_r
%37 ˿��ʵ�ʶ���(mm)��D_top_s 38 ˿��ʵ�ʸ���(mm)��D_bottom_s 39 ��ĸʵ�ʶ���(mm)��D_top_n
%40 ��ĸʵ�ʸ���(mm)��D_bottom_n 41 ����ʵ�ʶ���(mm)��D_top_r 42 ����ʵ�ʸ���(mm)��D_bottom_r
%43 ˿���о�(mm),D_pitch_s ,44 ��ĸ�о�(mm),D_pitch_n, 45 �����о�(mm)��D_pitch_r
%46 ˿���᳤��(mm)��length_shaft_s 47 ��ĸ�᳤��(mm)��length_shaft_n 48 �����᳤��(mm)��length_shaft_r 49 ˿�����⾶(mm)��diameter_outer_s 50 ��ĸ���⾶(mm)��diameter_outer_n 51 �������⾶(mm)��diameter_outer_r 52 ˿�����ھ�(mm)��diameter_inner_s 53 ��ĸ���ھ�(mm)��diameter_inner_n 54 �������ھ�(mm)��diameter_inner_r
%55 ˿�����Ƴ���(mm)��length_thread_s 56 ��ĸ���Ƴ���(mm)��length_thread_n 57 �������Ƴ���(mm)��length_thread_r 58 �����ϵ���Ч���Ƴ���(mm)��length_mesh_inner 59 �����ϵ���Ч���Ƴ���(mm)��length_mesh_outer 60 ��
%61 ����ֿ��(mm)��width_gear_left 62 �ҳ��ֿ��(mm)��width_gear_left 63 �󱣳ּܿ��(mm)��width_carrier_left 64 �󱣳ּܿ��(mm)��width_carrier_right
%65 ��ĸ����������˿������˵�λ��(mm)��delta_n_s 66 ��������������˿������˵�λ��(mm)��delta_r_s 67 ˿��������������˿������˵�λ��(mm),delta_thread_s 68 ��ĸ��������������ĸ����˵�λ��(mm),delta_thread_n 69 ���������������ڹ�������˵�λ��(mm),delta_thread_r 70 ����Ԥ���غ�(N)��preload_axial
%71 ˿������ܶ�(t/mm^-9)��density_s 72 ��ĸ����ܶ�(t/mm^-9)��density_n 73 ��������ܶ�(t/mm^-9)��density_r 74 ˿�ܵ���ģ��(Mpa)��E_s 75 ��ĸ����ģ��(Mpa)��E_n 76 ��������ģ��(Mpa)��E_r 77 ˿�ܲ��ɱȣ�possion_s 78 ��ĸ���ɱȣ�possion_n 79 �������ɱȣ�possion_r
%80 ������������ڹ��������λ��(mm)��delta_gear_left 81 �ҳ����������ڹ��������λ��(mm)��delta_gear_right 82 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_left 83 �����Ǽ��������ڹ��������λ��(mm)��delta_carrier_right
%84 �󱣳ּ��⾶(mm)��outer_carrier_left 85 �󱣳ּ��ھ�(mm)��inner_carrier_left 86 �ұ��ּ��⾶(mm)��outer_carrier_right 87 �ұ��ּ��ھ�(mm)��inner_carrier_right 88 ���ǹ���˿�Ƿ�ָ��Ч��(0��ʾ��ָ����1��ʾָ��),sign_efficiency 89 ��������Ч��(��Ԥ���غ�)��advance_efficiency 90 ��������Ч��(��Ԥ���غ�)��reverse_efficiency
%91 ���ַ���ģ��(mm)��m_n 92 �ڳ�Ȧ������z_r 93 ̫���ֳ�����z_s 94 �����ֳ�����z_p 95 ѹ����(rad)��press_angle 96 ����ѹ����(rad)��tran_press_angle 97 ������(rad)��helix_angle 98 �ڳ�Ȧ�����λϵ����n_r 99 ̫���ַ����λϵ����n_s 100 �����ַ����λϵ����n_p
%101 �������ľ�(mm)��work_center 102 ������ϵ�����Ͻǣ�mesh_angle(rad) 103 ������ϵ�����Ͻǣ�mesh_angle(rad) 104 �ڳ�Ȧ�ݶ�Բ��tran_ad_dia_r��mm�� 105 ̫���ֳݶ�Բ��tran_ad_dia_s��mm�� 106 �����ֳݶ�Բ��tran_ad_dia_p��mm�� 107 �ڳ�Ȧ�ݸ�Բ��tran_de_dia_r��mm�� 108 ̫���ֳݸ�Բ��tran_de_dia_s��mm�� 109 �����ֳݸ�Բ��tran_de_dia_p��mm��]
%110 �ڳ�Ȧ�Ľ�Բ��mm����tran_pitch_dia_r 111 ̫���ֵĽ�Բ��mm����tran_pitch_dia_s 112 �����ֵĽ�Բ��mm����tran_pitch_dia_p 113-120 ��
%121 ��ĸ������(mm)��width_web_n 122 ��ĸ�������ľ���ĸ����˾���(mm)��delta_web_n 123 ˿���������(mm)��width_left_web_s 124 ˿����������ľ�˿������˾���(mm)��delta_left_web_s 125 ˿���ҷ�����(mm)��width_right_web_s 126 ˿���ҷ������ľ�˿������˾���(mm)��delta_right_web_s 127-130 ��
%131 ��ĸ��Ե�����,shaft_self_nut 132 ˿����Ե����ţ�shaft_self_screw 133 ������Ե����ţ�shaft_self_roller,134-140 ��
%141 �����ֵ��߳ݶ���ϵ����142 �����ֵ��߶�϶ϵ����143 �����ֳݶ���������144 �ڳ�Ȧ���߳ݶ���ϵ����145 �ڳ�Ȧ���߶�϶ϵ����146 �ڳ�Ȧ�ݶ�������,147 ̫���ֵ��߳ݶ���ϵ����148 ̫���ֵ��߶�϶ϵ����149 ̫���ֳݶ�������
%150-160 �� 161 ��Ч�����������������ĸ����˵�λ��(mm)��thread_inner_nut 162 ��Ч��������������ڹ�������˵�λ��(mm)��thread_inner_roller 163 ��Ч���������������˿������˵�λ��(mm)��thread_outer_screw 164 ��Ч��������������ڹ�������˵�λ��(mm)��thread_outer_roller]
%%%����ʼ
%%%�������ǹ���˿����Ե�������ξ���roller_screw_segment��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
roller_screw_segment=zeros(0,16);
%%%��¼���Ϊorder_roller_screw�����ǹ���˿�ܵĻ�������
%���ǹ���˿����ţ�order_roller_screw
id_roller_screw=(roller_screw_initial(:,1)==order_roller_screw);
roller_screw_temp=roller_screw_initial(id_roller_screw,:);
%%%��¼���ǹ���˿�ܵ��ݾ�(mm)��pitch_screw
pitch_screw=roller_screw_temp(1,16);
%%%��¼˿��ʵ�ʸ���(mm)��D_bottom_s
D_bottom_s=roller_screw_temp(1,38);
%%%��¼��ĸʵ�ʸ���(mm)��D_bottom_n
D_bottom_n=roller_screw_temp(1,40);
%%%��¼����ʵ�ʸ���(mm)��D_bottom_r
D_bottom_r=roller_screw_temp(1,42);
%%%��¼˿���᳤��(mm)��length_shaft_s
length_shaft_s=roller_screw_temp(1,46);
%%%��¼��ĸ�᳤��(mm)��length_shaft_n
length_shaft_n=roller_screw_temp(1,47);
%%%��¼�����᳤��(mm)��length_shaft_r
length_shaft_r=roller_screw_temp(1,48);
%%%��¼��ĸ���⾶(mm)��diameter_outer_n
diameter_outer_n=roller_screw_temp(1,50);
%%%��¼˿�����ھ�(mm)��diameter_inner_s
diameter_inner_s=roller_screw_temp(1,52);
%%%��¼�������ھ�(mm)��diameter_inner_r
diameter_inner_r=roller_screw_temp(1,54);
%%%��¼�����ϵ���Ч���Ƴ���(mm)��length_mesh_inner
length_mesh_inner=roller_screw_temp(1,58);
%%%��¼�����ϵ���Ч���Ƴ���(mm)��length_mesh_outer
length_mesh_outer=roller_screw_temp(1,59);
%%%��¼��ĸ����������˿������˵�λ��(mm)��delta_n_s
delta_n_s=roller_screw_temp(1,65);
%%%��¼��������������˿������˵�λ��(mm)��delta_r_s
delta_r_s=roller_screw_temp(1,66);
%%%��¼˿��������������˿������˵�λ��(mm),delta_thread_s
delta_thread_s=roller_screw_temp(1,67);
%%%��¼��ĸ��������������ĸ����˵�λ��(mm),delta_thread_n
delta_thread_n=roller_screw_temp(1,68);
%%%��¼���������������ڹ�������˵�λ��(mm),delta_thread_r
delta_thread_r=roller_screw_temp(1,69);
%%%��¼˿������ܶ�(t/mm^-9)��density_s
density_s=roller_screw_temp(1,71);
%%%��¼��ĸ����ܶ�(t/mm^-9)��density_n
density_n=roller_screw_temp(1,72);
%%%��¼��������ܶ�(t/mm^-9)��density_r
density_r=roller_screw_temp(1,73);
%%%��¼˿�ܵ���ģ��(Mpa)��E_s
E_s=roller_screw_temp(1,74);
%%%��¼��ĸ����ģ��(Mpa)��E_n
E_n=roller_screw_temp(1,75);
%%%��¼��������ģ��(Mpa)��E_r
E_r=roller_screw_temp(1,76);
%%%��¼˿�ܲ��ɱȣ�possion_s
possion_s=roller_screw_temp(1,77);
%%%��¼��ĸ���ɱȣ�possion_n
possion_n=roller_screw_temp(1,78);
%%%��¼�������ɱȣ�possion_r
possion_r=roller_screw_temp(1,79);
%%%��¼��ĸ��Ե�����,shaft_self_nut
shaft_self_nut=roller_screw_temp(1,131);
%%%��¼˿����Ե�����,shaft_self_screw
shaft_self_screw=roller_screw_temp(1,132);
%%%��¼������Ե�����,shaft_self_roller
shaft_self_roller=roller_screw_temp(1,133);
%%%
%%%������Ч���Ƴ���(mm)��length_mesh
length_mesh=min(length_mesh_inner,length_mesh_outer);
%%%������Ч�������ϳ�����number
number=1+fix(length_mesh/pitch_screw);
%%%����˿��������������˿������˵�λ��(mm)��thread_left_s
thread_left_s=delta_thread_s;
%%%�������������������˿������˵�λ��(mm)��thread_left_r
thread_left_r=delta_thread_r+delta_r_s;
%%%������ĸ������������˿������˵�λ��(mm)��thread_left_n
thread_left_n=delta_thread_n+delta_n_s;
%%%������ĸ��Ч������˾���ĸ����˵ľ���(mm)��distance_nut_left
distance_nut_left=max([thread_left_n,thread_left_s,thread_left_r])-delta_n_s;
%%%������ĸ��Ч�����Ҷ˾���ĸ����˵ľ���(mm)��distance_nut_right
distance_nut_right=distance_nut_left+(number-1)*pitch_screw;
%%%����˿����Ч������˾�˿������˵ľ���(mm)��distance_screw_left
distance_screw_left=max([thread_left_n,thread_left_s,thread_left_r]);
%%%����˿����Ч�����Ҷ˾�˿������˵ľ���(mm)��distance_screw_right
distance_screw_right=distance_screw_left+(number-1)*pitch_screw;
%%%���������Ч������˾��������˵ľ���(mm)��distance_roller_left
distance_roller_left=max([thread_left_n,thread_left_s,thread_left_r])-delta_r_s;
%%%���������Ч�����Ҷ˾��������˵ľ���(mm)��distance_roller_right
distance_roller_right=distance_roller_left+(number-1)*pitch_screw;
%%%
%%%������ĸ��Ե�������ξ���shaft_segment_nut��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
shaft_segment_nut=zeros(0,16);
for kk=1:number+1 %��Ч�������ϳ�����number
    if kk==1 %��ĸ���������ǰ��
        shaft_segment_nut(kk,1)=shaft_self_nut;  %��ĸ��Ե�����,shaft_self_nut
        shaft_segment_nut(kk,2)=kk;  %��������к�
        shaft_segment_nut(kk,3)=distance_nut_left; %��ĸ��Ч������˾���ĸ����˵ľ���(mm)��distance_nut_left
        shaft_segment_nut(kk,4)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,5)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
        shaft_segment_nut(kk,6)=E_n;  %��ĸ����ģ��(Mpa)��E_n
        shaft_segment_nut(kk,7)=possion_n;  %��ĸ���ɱȣ�possion_n
        shaft_segment_nut(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_nut(kk,9)=distance_nut_left;  %��ĸ��Ч������˾���ĸ����˵ľ���(mm)��distance_nut_left
        shaft_segment_nut(kk,10)=density_n;  %��ĸ����ܶ�(t/mm^-9)��density_n
        shaft_segment_nut(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_nut(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_nut(kk,13)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,14)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,15)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
        shaft_segment_nut(kk,16)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
    elseif kk==number+1 %��ĸ���������ǰ��
        shaft_segment_nut(kk,1)=shaft_self_nut;  %��ĸ��Ե�����,shaft_self_nut
        shaft_segment_nut(kk,2)=kk;  %��������к�
        shaft_segment_nut(kk,3)=length_shaft_n-distance_nut_right; %��ĸ��Ч�����Ҷ˾���ĸ����˵ľ���(mm)��distance_nut_right
        shaft_segment_nut(kk,4)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,5)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
        shaft_segment_nut(kk,6)=E_n;  %��ĸ����ģ��(Mpa)��E_n
        shaft_segment_nut(kk,7)=possion_n;  %��ĸ���ɱȣ�possion_n
        shaft_segment_nut(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_nut(kk,9)=length_shaft_n;  %��ĸ�᳤��(mm)��length_shaft_n
        shaft_segment_nut(kk,10)=density_n;  %��ĸ����ܶ�(t/mm^-9)��density_n
        shaft_segment_nut(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_nut(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_nut(kk,13)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,14)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,15)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
        shaft_segment_nut(kk,16)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
    else
        shaft_segment_nut(kk,1)=shaft_self_nut;  %��ĸ��Ե�����,shaft_self_nut
        shaft_segment_nut(kk,2)=kk;  %��������к�
        shaft_segment_nut(kk,3)=pitch_screw; %���ǹ���˿�ܵ��ݾ�(mm)��pitch_screw
        shaft_segment_nut(kk,4)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,5)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
        shaft_segment_nut(kk,6)=E_n;  %��ĸ����ģ��(Mpa)��E_n
        shaft_segment_nut(kk,7)=possion_n;  %��ĸ���ɱȣ�possion_n
        shaft_segment_nut(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_nut(kk,9)=distance_nut_left+(kk-1)*pitch_screw;  %��ĸ�᳤��(mm)��length_shaft_n
        shaft_segment_nut(kk,10)=density_n;  %��ĸ����ܶ�(t/mm^-9)��density_n
        shaft_segment_nut(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_nut(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_nut(kk,13)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,14)=diameter_outer_n;  %��ĸ���⾶(mm)��diameter_outer_n
        shaft_segment_nut(kk,15)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
        shaft_segment_nut(kk,16)=D_bottom_n;  %��ĸʵ�ʸ���(mm)��D_bottom_n
    end
end
%%%��¼���ǹ���˿����Ե�������ξ���roller_screw_segment��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
roller_screw_segment(1:kk,:)=shaft_segment_nut;
%%%�����������dd
dd=kk;
%%%
%%%����˿����Ե�������ξ���shaft_segment_screw��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
shaft_segment_screw=zeros(0,16);
for kk=1:number+1 %��Ч�������ϳ�����number
    if kk==1 %˿�����������ǰ��
        shaft_segment_screw(kk,1)=shaft_self_screw;  %˿����Ե�����,shaft_self_screw
        shaft_segment_screw(kk,2)=kk;  %��������к�
        shaft_segment_screw(kk,3)=distance_screw_left; %˿����Ч������˾�˿������˵ľ���(mm)��distance_screw_left
        shaft_segment_screw(kk,4)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,5)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
        shaft_segment_screw(kk,6)=E_s;  %˿�ܵ���ģ��(Mpa)��E_s
        shaft_segment_screw(kk,7)=possion_s;  %˿�ܲ��ɱȣ�possion_s
        shaft_segment_screw(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_screw(kk,9)=distance_screw_left;  %˿����Ч������˾�˿������˵ľ���(mm)��distance_screw_left
        shaft_segment_screw(kk,10)=density_s;  %˿������ܶ�(t/mm^-9)��density_s
        shaft_segment_screw(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_screw(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_screw(kk,13)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,14)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,15)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
        shaft_segment_screw(kk,16)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
    elseif kk==number+1 %˿�����������ǰ��
        shaft_segment_screw(kk,1)=shaft_self_screw;  %˿����Ե�����,shaft_self_screw
        shaft_segment_screw(kk,2)=kk;  %��������к�
        shaft_segment_screw(kk,3)=length_shaft_s-distance_screw_right; %˿����Ч�����Ҷ˾�˿������˵ľ���(mm)��distance_screw_right
        shaft_segment_screw(kk,4)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,5)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
        shaft_segment_screw(kk,6)=E_s;  %˿�ܵ���ģ��(Mpa)��E_s
        shaft_segment_screw(kk,7)=possion_s;  %˿�ܲ��ɱȣ�possion_s
        shaft_segment_screw(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_screw(kk,9)=length_shaft_s;  %˿���᳤��(mm)��length_shaft_s
        shaft_segment_screw(kk,10)=density_s;  %˿������ܶ�(t/mm^-9)��density_s
        shaft_segment_screw(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_screw(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_screw(kk,13)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,14)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,15)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
        shaft_segment_screw(kk,16)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
    else
        shaft_segment_screw(kk,1)=shaft_self_screw;  %˿����Ե�����,shaft_self_screw
        shaft_segment_screw(kk,2)=kk;  %��������к�
        shaft_segment_screw(kk,3)=pitch_screw; %���ǹ���˿�ܵ��ݾ�(mm)��pitch_screw
        shaft_segment_screw(kk,4)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,5)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
        shaft_segment_screw(kk,6)=E_s;  %˿�ܵ���ģ��(Mpa)��E_s
        shaft_segment_screw(kk,7)=possion_s;  %˿�ܲ��ɱȣ�possion_s
        shaft_segment_screw(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_screw(kk,9)=distance_screw_left+(kk-1)*pitch_screw;  %˿���᳤��(mm)��length_shaft_s
        shaft_segment_screw(kk,10)=density_s;  %˿������ܶ�(t/mm^-9)��density_s
        shaft_segment_screw(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_screw(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_screw(kk,13)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,14)=D_bottom_s;  %˿��ʵ�ʸ���(mm)��D_bottom_s
        shaft_segment_screw(kk,15)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
        shaft_segment_screw(kk,16)=diameter_inner_s;  %˿�����ھ�(mm)��diameter_inner_s
    end
end
%%%��¼���ǹ���˿����Ե�������ξ���roller_screw_segment��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
roller_screw_segment(dd+1:dd+kk,:)=shaft_segment_screw;
%%%���¼�������dd
dd=dd+kk;
%%%
%%%���ɹ�����Ե�������ξ���shaft_segment_roller��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
shaft_segment_roller=zeros(0,16);
for kk=1:number+1 %��Ч�������ϳ�����number
    if kk==1 %�������������ǰ��
        shaft_segment_roller(kk,1)=shaft_self_roller;  %������Ե�����,shaft_self_roller
        shaft_segment_roller(kk,2)=kk;  %��������к�
        shaft_segment_roller(kk,3)=distance_roller_left; %������Ч������˾��������˵ľ���(mm)��distance_roller_left
        shaft_segment_roller(kk,4)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,5)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
        shaft_segment_roller(kk,6)=E_r;  %��������ģ��(Mpa)��E_r
        shaft_segment_roller(kk,7)=possion_r;  %�������ɱȣ�possion_r
        shaft_segment_roller(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_roller(kk,9)=distance_roller_left;  %������Ч������˾��������˵ľ���(mm)��distance_roller_left
        shaft_segment_roller(kk,10)=density_r;  %��������ܶ�(t/mm^-9)��density_r
        shaft_segment_roller(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_roller(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_roller(kk,13)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,14)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,15)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
        shaft_segment_roller(kk,16)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
    elseif kk==number+1 %�������������ǰ��
        shaft_segment_roller(kk,1)=shaft_self_roller;  %������Ե�����,shaft_self_roller
        shaft_segment_roller(kk,2)=kk;  %��������к�
        shaft_segment_roller(kk,3)=length_shaft_r-distance_roller_right; %������Ч�����Ҷ˾��������˵ľ���(mm)��distance_roller_right
        shaft_segment_roller(kk,4)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,5)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
        shaft_segment_roller(kk,6)=E_r;  %��������ģ��(Mpa)��E_r
        shaft_segment_roller(kk,7)=possion_r;  %�������ɱȣ�possion_r
        shaft_segment_roller(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_roller(kk,9)=length_shaft_r;  %�����᳤��(mm)��length_shaft_r
        shaft_segment_roller(kk,10)=density_r;  %��������ܶ�(t/mm^-9)��density_r
        shaft_segment_roller(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_roller(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_roller(kk,13)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,14)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,15)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
        shaft_segment_roller(kk,16)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
    else
        shaft_segment_roller(kk,1)=shaft_self_roller;  %������Ե�����,shaft_self_roller
        shaft_segment_roller(kk,2)=kk;  %��������к�
        shaft_segment_roller(kk,3)=pitch_screw; %���ǹ����������ݾ�(mm)��pitch_screw
        shaft_segment_roller(kk,4)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,5)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
        shaft_segment_roller(kk,6)=E_r;  %��������ģ��(Mpa)��E_r
        shaft_segment_roller(kk,7)=possion_r;  %�������ɱȣ�possion_r
        shaft_segment_roller(kk,8)=0.886;  %8 ����μ��б�������
        shaft_segment_roller(kk,9)=distance_roller_left+(kk-1)*pitch_screw;  %�����᳤��(mm)��length_shaft_r
        shaft_segment_roller(kk,10)=density_r;  %��������ܶ�(t/mm^-9)��density_r
        shaft_segment_roller(kk,11)=1;  %11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У�
        shaft_segment_roller(kk,12)=0;  %12 ������ҽڵ�װ������
        shaft_segment_roller(kk,13)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,14)=D_bottom_r;  %����ʵ�ʸ���(mm)��D_bottom_r
        shaft_segment_roller(kk,15)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
        shaft_segment_roller(kk,16)=diameter_inner_r;  %�������ھ�(mm)��diameter_inner_r
    end
end
%%%��¼���ǹ���˿����Ե�������ξ���roller_screw_segment��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
roller_screw_segment(dd+1:dd+kk,:)=shaft_segment_roller;
%%%���¼�������dd
dd=dd+kk;
%%%
%%%������ĸնȾ���
for ii=1:dd %��������dd
    poisson=roller_screw_segment(ii,7);
    diameter_outer=roller_screw_segment(ii,4);
    diameter_inner=roller_screw_segment(ii,5);
    diameter_ratio=diameter_inner/diameter_outer;
    a_factor=6*(1+poisson)*(1+diameter_ratio^2)^2/((7+6*poisson)*(1+diameter_ratio^2)^2+(20+12*poisson)*diameter_ratio^2);
    roller_screw_segment(ii,8)=a_factor;
end
end
%%%�������

%%%�������ǹ���˿�ܵ���������ѹ��Ťת��ȫ������ϵ�ĸնȾ���shaft_stiffness
function [roller_screw_stiffness]=roller_screw_stiffness_produce(order_roller_screw,roller_screw_segment)
%%%�������
%���ǹ���˿�ܵ������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���roller_screw_stiffness��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
%%%��������
%���ǹ���˿�ܵ���ţ�order_roller_screw
%��ϵ���������β�������roller_screw_segment��[1 ������ 2 ��������к� 3 ����γ���(mm) 4 �����ƽ���⾶(mm) 5 �����ƽ���ھ�(mm) 6 ����ε���ģ��(Mpa) 7 ����β��ɱ� 8 ����μ��б������� 9 ������Ҷ˾�����˾���(mm) 10 ������ܶ�(t/mm3) 11 ������ҽڵ��Ƿ�Ϊ���нڵ㣨0��ʾ�ǹ��� 1��ʾ���У� 12 ������ҽڵ�װ����� 13 ���������⾶(mm) 14 ������Ҷ��⾶(mm) 15 ���������ھ�(mm) 16 ������Ҷ�(mm) ]
%%%����ʼ
%%%�������ǹ���˿�ܵ������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���roller_screw_stiffness��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
roller_screw_stiffness=zeros(0,1);
%%%�����������dd
dd=0;
%%%
for mm=1:3
    %%%������ĸ/˿��/��������Ե�����ţ�order_shaft
    if mm~=3
        %%%������ĸ/˿�ܵ���Ե�����ţ�order_shaft
        order_shaft=fix(order_roller_screw)+0.0001*mm; %���ǹ���˿�ܵ���ţ�order_roller_screw
    elseif mm==3
        %%%�����ݹ�������Ե�����ţ�order_shaft
        order_shaft=fix(order_roller_screw)+0.0001*(mm+1); %���ǹ���˿�ܵ���ţ�order_roller_screw
    end
    %��¼���Ϊorder_roller_screw�������ξ���shaft_segment_single����1 ��������к� 2 ����γ���(mm) 3 �����ƽ���⾶(mm) 4 �����ƽ���ھ�(mm) 5 ����ε���ģ��(Mpa) 6 ����β��ɱ� 7 ����μ��б������� 8 ������Ҷ˾�����˾���(mm) 9 ������ܶ�(t/mm3)��
    id_shaft_segment=(roundn(roller_screw_segment(:,1),-9)==roundn(order_shaft,-9));
    shaft_segment_single=roller_screw_segment(id_shaft_segment,2:10);
    %�����������������е����������ξ���
    shaft_segment_single=sortrows(shaft_segment_single,1);
    %�������Ϊorder_shaft��Ľڵ�����shaft_segment
    shaft_segment=sum(id_shaft_segment)+1;
    %%%�������Ϊorder_shaft������������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���shaft_single_stiffness��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    shaft_single_stiffness=zeros(6*shaft_segment,6*shaft_segment);
    %�������Ϊorder_shaft������������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���shaft_single_stiffness_local��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness_local*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    shaft_single_stiffness_local=zeros(6*shaft_segment,6*shaft_segment);
    %%%�������ε������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���
    for k=1:shaft_segment
        if k==1 %��1������
            %�����񶯸ն�
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
            %�����񶯸ն�
            axial_k=E*area/L_shaft;
            shaft_single_stiffness_local(6*k-3,6*k-3)=axial_k;
            shaft_single_stiffness_local(6*k-3,6*k+3)=-axial_k;
            %Ťת�񶯸ն�
            Ip=pi*(shaft_segment_single(k,3)^4-shaft_segment_single(k,4)^4)/32;
            stiffness_twist=G*Ip/L_shaft;
            shaft_single_stiffness_local(6*k,6*k)=stiffness_twist;
            shaft_single_stiffness_local(6*k,6*k+6)=-1*stiffness_twist;
        elseif k==shaft_segment %���һ������
            %�����񶯸ն�
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
            %�����񶯸ն�
            axial_k=E*area/L_shaft;
            shaft_single_stiffness_local(6*k-3,6*k-9)=-axial_k;
            shaft_single_stiffness_local(6*k-3,6*k-3)=axial_k;
            %Ťת�񶯸ն�
            Ip=pi*(shaft_segment_single(k-1,3)^4-shaft_segment_single(k-1,4)^4)/32;
            stiffness_twist=G*Ip/L_shaft;
            shaft_single_stiffness_local(6*k,6*k-6)=-stiffness_twist;
            shaft_single_stiffness_local(6*k,6*k)=stiffness_twist;
        else  %�м����
            %�����񶯸ն�
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
            %�����񶯸ն�
            axial_k_1=E_1*area_1/L_shaft_1;
            axial_k_2=E_2*area_2/L_shaft_2;
            shaft_single_stiffness_local(6*k-3,6*k-9)=-axial_k_1;
            shaft_single_stiffness_local(6*k-3,6*k-3)=axial_k_1+axial_k_2;
            shaft_single_stiffness_local(6*k-3,6*k+3)=-axial_k_2;
            %Ťת�񶯸ն�
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
    %%%������������񶯡������񶯡�Ťת�񶯵���ʱ�նȾ���shaft_single_stiffness_temp��ת���ɵ�����������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ��󣬣�shaft_single_stiffness��
    %������������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���shaft_single_stiffness��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    %������������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���shaft_single_stiffness_local��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness_local*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    %���嵥����������񶯡������񶯡�Ťת�񶯵���ʱ�նȾ���shaft_single_stiffness_temp
    %������
    shaft_single_stiffness_temp=zeros(6*shaft_segment,6*shaft_segment);
    for ii=1:shaft_segment  %���Ϊorder_shaft��Ľڵ�����shaft_segment
        for jj=1:shaft_segment  %���Ϊorder_shaft��Ľڵ�����shaft_segment
            %������
            shaft_single_stiffness_temp(6*(ii-1)+1,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+1,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+2,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+4,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+3,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+3,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+4,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+5,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+5,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+2,6*(jj-1)+1:6*jj);
            shaft_single_stiffness_temp(6*(ii-1)+6,6*(jj-1)+1:6*jj)=shaft_single_stiffness_local(6*(ii-1)+6,6*(jj-1)+1:6*jj);
            %������
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+1)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+1);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+2)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+4);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+3)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+3);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+4)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+5);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+5)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+2);
            shaft_single_stiffness(6*(ii-1)+1:6*ii,6*(jj-1)+6)=shaft_single_stiffness_temp(6*(ii-1)+1:6*ii,6*(jj-1)+6);
        end
    end
    %%%
    %%%������ĸ/˿��/��������Ե��Եصĸն�
    addition_stiffness=zeros(6,6);
    addition_stiffness(1,1)=10^3;
    addition_stiffness(2,2)=10^3;
    addition_stiffness(3,3)=10^3;
    addition_stiffness(4,4)=10^5;
    addition_stiffness(5,5)=10^5;
    addition_stiffness(6,6)=10^5;
    shaft_single_stiffness(1:6,1:6)=addition_stiffness;
    %%%��¼���ǹ���˿�ܵ������񶯡������񶯡�Ťת�񶯵ľֲ�����ϵ�ĸնȾ���roller_screw_stiffness��[Fx(N) Fy(N) Fz(N) Mx(Nmm) My(Nmm) Tz(Nmm)]'=shaft_single_stiffness*[delta_x(mm) delta_y(mm) delta_z(mm) theta_x(rad) theta_y(rad) theta_z(rad)]'
    roller_screw_stiffness(dd+1:dd+6*shaft_segment,1)=order_shaft; %��ĸ/˿��/������Ե������ţ�order_shaft
    roller_screw_stiffness(dd+1:dd+6*shaft_segment,2:1+6*shaft_segment)=shaft_single_stiffness;
    %%%���¼�������dd
    dd=dd+6*shaft_segment;
end
end
%%%�������