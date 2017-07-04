clear,clc
%% 数据结构
% dd为PML层宽度（最外层）,坐标为[-dd，0.07+dd]*[-dd，0.064+dd]
% x1y1为空气区域（第二层）,坐标为[0，0.07]*[0，0.064]
% x2y2为超材料介质（最内层），,坐标为[0.024，0.054]*[0.002，0.062]
% 直线的坐标为0.004*[0.025，0.035]

h=2.0e-4;
dd=12*h;
x1min=0;x1max=0.07;y1min=0;y1max=0.064;
x2min=0.024;x2max=0.054;y2min=0.002;y2max=0.062;
x3=0.004;y3min=0.025;y3max=0.035;

%% g1=PML边界区域 g2=超材料区域 g3=空气介质区域
g1=[
   2 x1min-dd x1max+dd y1min-dd y1min-dd 1 0
   2 x1max+dd x1max+dd y1min-dd y1max+dd 1 0
   2 x1max+dd x1min-dd y1max+dd y1max+dd 1 0
   2 x1min-dd x1min-dd y1max+dd y1min-dd 1 0
   2 x1min x1max y1min y1min 1 0
   2 x1max x1max y1min y1max 1 0
   2 x1max x1min y1max y1max 1 0
   2 x1min x1min y1max y1min 1 0
   ]';
g3=[
   2 x1min x1max y1min y1min 1 0
   2 x1max x1max y1min y1max 1 0
   2 x1max x1min y1max y1max 1 0
   2 x1min x1min y1max y1min 1 0
   2 x2min x2max y2min y2min 1 0
   2 x2max x2max y2min y2max 1 0
   2 x2max x2min y2max y2max 1 0
   2 x2min x2min y2max y2min 1 0
   ]';
g2=[
   2 x2min x2max y2min y2min 1 0
   2 x2max x2max y2min y2max 1 0
   2 x2max x2min y2max y2max 1 0
   2 x2min x2min y2max y2min 1 0
   ]';
%% 分别对三个区域进行网格剖分，然后再拼接在一起
[no2xy1,e1,EToV1] = initmesh(g1,'hmax',10*h);
[no2xy2,e2,EToV2] = initmesh(g2,'hmax',10*h);
[no2xy3,e3,EToV3] = initmesh(g3,'hmax',10*h);

nn1=size(no2xy1,2);
nn2=size(no2xy2,2);

no2xy=[no2xy1 no2xy2 no2xy3];
e=[e1 e2+nn1 e3+nn1+nn2];
EToV=[EToV1 EToV2+nn1 EToV3+nn1+nn2];%画图用

%% 单独查看某个区域的网格情况
% no2xy=no2xy1;
% e=e1;
% EToV=EToV1;
%% 画图
pdemesh(no2xy,e,EToV)%画图
