import numpy as np

class RB_IGCT_Doping():
    '''
    对应于 RB_IGCT 网格的掺杂函数
    '''
    def __init__(self):
        pass
    
    def __call__(self,p):
        return self.TotalDoping(p)

    def Erfc(self,x):
        a = [-1.26551223,1.00002368,0.37409196,0.09678418,-0.18628806,0.27886807,
            -1.13520398,1.48851587,-0.82215223,0.17087277]
        v = np.ones(len(x))
        z = np.fabs(x)

        t = 1.0/(1.0+0.5*z)
        flag1 = (z>0)
        t1 = t[flag1]
        z1 = z[flag1]
        v[flag1]=t1*np.exp((-z1*z1)+a[0]+t1*(a[1]+t1*(a[2]+t1*(a[3]+t1*(a[4]+t1*(a[5]+
            t1*(a[6]+t1*(a[7]+t1*(a[8]+t1*a[9])))))))));
        v[x<0]=2.0-v[x<0]
        return v

    def Erf(self,x):
        return 1.0 - self.Erfc(x)

    def Gaussian(self,x):
        return np.exp(-x*x)

    def TotalDoping(self,node):
        # 传入空间坐标，传出总掺杂浓度
        # 逐步工艺叠加，分别计算 Nd 和 Na，最后相减得到总掺杂浓度
        base = 1.0
        x = node[:,0]
        y = node[:,1]

        peak_n_emitter = 1.2e20
        peak_p_plus_base = 5e17
        peak_p_base_cathode = 5e14
        peak_p_base_anode = 2e14
        constant_n_base = 9.8e12
        peak_p_emitter = 5e18

        H_total = 1500.0/base
        W_total = 245.0/base

        # 两种杂质浓度
        NN = len(node)
        Nd = 1e-100*np.ones(NN,dtype=np.float64)
        Na = 1e-100*np.ones(NN,dtype=np.float64)

        # 距离掺杂 baseline 的距离，用于计算该位置的 profile。
        distance_to_baseline = 0.0
        # 大家都要有 n base 的基础掺杂
        Nd += constant_n_base

       # 顶部的 n emitter，通过 error function 添加
        n_emitter_erf_length = 3.6 / base # erf 的 length
        n_emitter_symmetric_position = 11.0 / base # 对称点
        L_n_emitter = 110.0 / base # 未被挖槽的阴极凸台长度
        H_slot = 18.0 / base # 凸台挖槽深度
        L_n_emitter_baseline = L_n_emitter - H_slot # 扩散的 baseline 的长度
        lateral_factor = 1.0 # 横向的 factor

        down_baseline_flag = (node[:,0]<=L_n_emitter_baseline) # 在扩散 baseline 的正下方
        right_baseline_flag = ~down_baseline_flag # 在扩散 baseline 的右侧

        # 在扩散 baseline 的正下方
        distance_to_baseline = node[down_baseline_flag,1]
        Nd[down_baseline_flag] += peak_n_emitter/2.0*(1.0+self.Erf((n_emitter_symmetric_position-distance_to_baseline)/n_emitter_erf_length))
        # 若在扩散 baseline 的右侧, 横向使用 error function分布
        distance_to_baseline = node[right_baseline_flag,1]
        doping_vertical = peak_n_emitter/2.0*(1.0+self.Erf((n_emitter_symmetric_position-distance_to_baseline) / n_emitter_erf_length))
        std_dev_y = n_emitter_erf_length/np.sqrt(2.0)
        std_dev_x = lateral_factor*std_dev_y;
        Nd[right_baseline_flag] +=doping_vertical*np.exp(-0.5*(node[right_baseline_flag,0]-L_n_emitter_baseline)*(node[right_baseline_flag,0]-L_n_emitter_baseline)/std_dev_x/std_dev_x);

        # 顶部的 p+ base，通过 gaussian function 添加
        p_plus_base_length = 23.0 / base;
        distance_to_baseline = node[:,1];
        Na += peak_p_plus_base*self.Gaussian(distance_to_baseline/p_plus_base_length)

        # 底部的 p base，通过 gaussian function 添加
        p_base_length = 69.0/base
        distance_to_baseline = H_total - y
        Na += peak_p_base_anode*self.Gaussian(distance_to_baseline/p_base_length)


        # 底部的 p emitter，通过 gaussian function 添加
        p_emitter_length = 6.1/base
        distance_to_baseline = H_total - y;
        Na += peak_p_emitter*self.Gaussian(distance_to_baseline/p_emitter_length);
        return np.abs(Nd-Na)/1e20

class BJT_doping():
    '''
    矩形BJT器件的掺杂函数
    '''
    def __init__(self):
        pass
    
    def __call__(self,p):
        return TotalDoping(p)

    def Erfc(x):
        a = [-1.26551223,1.00002368,0.37409196,0.09678418,-0.18628806,0.27886807,
            -1.13520398,1.48851587,-0.82215223,0.17087277]
        v = np.ones(len(x))
        z = np.fabs(x)

        t = 1.0/(1.0+0.5*z)
        flag1 = (z>0)
        t1 = t[flag1]
        z1 = z[flag1]
        v[flag1]=t1*np.exp((-z1*z1)+a[0]+t1*(a[1]+t1*(a[2]+t1*(a[3]+t1*(a[4]+t1*(a[5]+
            t1*(a[6]+t1*(a[7]+t1*(a[8]+t1*a[9])))))))));
        v[x<0]=2.0-v[x<0]
        return v

    def Erf(x):
        return 1.0 - Erfc(x)

    def Gaussian(x):
        return np.exp(-x*x)

    def TotalDoping(node):
        # 传入空间坐标，传出总掺杂浓度
        # 逐步工艺叠加，分别计算 Nd 和 Na，最后相减得到总掺杂浓度
        x = node[:,0]
        y = node[:,1]
        
        constant_n_base = 9e13
        peak_p_base = 1e18
        peak_n_emitter = 1e20
        peak_n_collector = 4e19

        H_total = 220
        W_total = 100

        # 两种杂质浓度
        NN = len(node)
        Nd = 1e-100*np.ones(NN,dtype=np.float64)
        Na = 1e-100*np.ones(NN,dtype=np.float64)

        # 距离掺杂 baseline 的距离，用于计算该位置的 profile。
        distance_to_baseline = 0.0

        # 大家都要有 n base 的基础掺杂
        Nd += constant_n_base

        # 顶部的 n emitter，通过 error function 添加
        n_emitter_erf_length = 3.6  # erf 的 length
        n_emitter_symmetric_position = 11.0 # 对称点
        lateral_factor = 1.0 # 横向的 factor

        down_baseline_flag1 = (node[:,0]<=65) # 在扩散 baseline 的正下方
        down_baseline_flag2 = (node[:,0]>=35)
        down_baseline_flag = down_baseline_flag1 & down_baseline_flag2
        right_baseline_flag = (node[:,0]>65) # 在扩散 baseline 的右侧
        left_baseline_flag  = (node[:,0]<35) #在扩散 baseline 的左侧

        # 在扩散 baseline 的正下方
        distance_to_baseline = node[down_baseline_flag,1]
        Nd[down_baseline_flag] += peak_n_emitter/2.0*(1.0+Erf((n_emitter_symmetric_position-distance_to_baseline)/n_emitter_erf_length))

        # 若在扩散 baseline 的右侧, 横向使用 error function分布
        distance_to_baseline = node[right_baseline_flag,1]
        doping_vertical = peak_n_emitter/2.0*(1.0+Erf((n_emitter_symmetric_position-distance_to_baseline) / n_emitter_erf_length))
        std_dev_y = n_emitter_erf_length/np.sqrt(2.0)
        std_dev_x = lateral_factor*std_dev_y;
        Nd[right_baseline_flag] += doping_vertical*np.exp(-0.5*(node[right_baseline_flag,0]-65)*(node[right_baseline_flag,0]-65)/std_dev_x/std_dev_x);

        # 若在扩散 baseline 的左侧，横向使用 error function分布
        distance_to_baseline = node[right_baseline_flag,1]
        doping_vertical = peak_n_emitter/2.0*(1.0+Erf((n_emitter_symmetric_position-distance_to_baseline) / n_emitter_erf_length))
        std_dev_y = n_emitter_erf_length/np.sqrt(2.0)
        std_dev_x = lateral_factor*std_dev_y;
        Nd[left_baseline_flag] += doping_vertical*np.exp(-0.5*(node[right_baseline_flag,0]-35)*(node[right_baseline_flag,0]-35)/std_dev_x/std_dev_x);

        # 底部的 p base，通过 gaussian function 添加
        p_base_length = 50.0
        distance_to_baseline = node[:,1]
        Na += peak_p_base_anode*Gaussian(distance_to_baseline/p_base_length)

        # 底部的 n collector,通过 gaussian function 添加
        p_emitter_length = 35.0
        distance_to_baseline = H_total - node[:,1];
        Nd += peak_n_emitter*Gaussian(distance_to_baseline/n_collector_length);
        return np.abs(Nd-Na)/1e20

