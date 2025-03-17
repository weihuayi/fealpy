import numpy as np

def load_roller_screw_1(Fn, force_axial, pitch_screw, number_roller, ratio_nut, ratio_screw,
                        E_n, E_s, E_r, area_n, area_s, area_r, C_delta_inner, C_delta_outer,
                        E0_inner, E0_outer, sum_p_inner, sum_p_outer, flexible_inner, flexible_outer,
                        modify_nut, modify_screw, share):

    # 计算法向载荷个数
    Fn = Fn.reshape(62, 5)
    number = Fn.shape[0] // 2  
    
    # 初始化右侧常数项矩阵
    FF = np.zeros((2 * number, number_roller))
    
    # 转换为毫米制单位系数
    unit_scale = 1e3  
    
    # 遍历每个滚柱构建方程组
    for kk in range(number_roller):  
        # 螺母侧方程构建
        for ii in range(number):     
            if ii == 0:  
                FF[ii, kk] = np.sum(np.maximum(0, Fn[:number, kk])) - share[kk, 0] * force_axial / ratio_nut
            else:        # ii >= 1的情况
                # 提取相关子矩阵并计算
                Fn_sub = np.maximum(0, Fn[ii:number, :number_roller])
                sum_term = np.sum(Fn_sub) * ratio_nut * pitch_screw / (area_n * E_n)
                
                # 接触变形项
                F_prev = np.maximum(0, Fn[ii-1, kk])
                F_curr = np.maximum(0, Fn[ii, kk])
                contact_term = (C_delta_inner * E0_inner**2 * sum_p_inner**(1/3) *
                               (F_prev**(2/3) - F_curr**(2/3)) / ratio_nut)
                
                # 柔度项
                flexible_term = flexible_inner * (F_prev - F_curr)
                
                # 修形项
                modify_term = (modify_nut[ii-1, kk] - modify_nut[ii, kk])
                
                # 滚柱平衡项
                nut_sum = np.sum(np.maximum(0, Fn[ii:number, kk])) * ratio_nut
                screw_sum = np.sum(np.maximum(0, Fn[ii+number : 2*number, kk])) * ratio_screw
                balance_term = (nut_sum - screw_sum) * pitch_screw / (area_r * E_r)
                
                # 组合所有项
                FF[ii, kk] = unit_scale * (sum_term - contact_term - flexible_term - modify_term + balance_term)
        
        # 丝杠侧方程构建
        for ii in range(number):  
            if ii == 0:  
                FF[ii+number, kk] = np.sum(np.maximum(0, Fn[number : 2*number, kk])) - share[kk, 0] * force_axial / ratio_screw
            else:
                # 提取相关子矩阵
                Fn_sub = np.maximum(0, Fn[ii+number : 2*number, :number_roller])
                sum_term = np.sum(Fn_sub) * ratio_screw * pitch_screw / (area_s * E_s)
                
                # 接触变形项
                F_prev = np.maximum(0, Fn[ii-1+number, kk])
                F_curr = np.maximum(0, Fn[ii+number, kk])
                contact_term = (C_delta_outer * E0_outer**2 * sum_p_outer**(1/3) *
                              (F_prev**(2/3) - F_curr**(2/3)) / ratio_screw)
                
                # 柔度项
                flexible_term = flexible_outer * (F_prev - F_curr)
                
                # 修形项
                modify_term = (modify_screw[ii-1, kk] - modify_screw[ii, kk])
                
                # 滚柱平衡项（注意符号变化）
                nut_sum = np.sum(np.maximum(0, Fn[ii:number, kk])) * ratio_nut
                screw_sum = np.sum(np.maximum(0, Fn[ii+number : 2*number, kk])) * ratio_screw
                balance_term = (nut_sum - screw_sum) * pitch_screw / (area_r * E_r)
                
                # 组合所有项
                FF[ii+number, kk] = unit_scale * (sum_term - contact_term - flexible_term - modify_term - balance_term)
    
    return FF.flatten()

import numpy as np

def load_roller_screw_2(Fn, force_axial, pitch_screw, number_roller, ratio_nut, ratio_screw,
                        E_n, E_s, E_r, area_n, area_s, area_r, C_delta_inner, C_delta_outer,
                        E0_inner, E0_outer, sum_p_inner, sum_p_outer, flexible_inner, flexible_outer,
                        modify_nut, modify_screw, share):

    # 计算法向载荷个数
    Fn.reshape(62, 5)
    number = Fn.shape[0] // 2
    
    # 初始化右侧常数项矩阵
    FF = np.zeros((2 * number, number_roller))
    
    # 单位转换系数
    unit_scale = 1e3  # 与MATLAB代码中的10^3对应

    for kk in range(number_roller):
        # 螺母侧方程构建（与load_roller_screw_1相同）
        for ii in range(number):
            if ii == 0:
                FF[ii, kk] = np.sum(np.maximum(0, Fn[:number, kk])) - share[kk, 0] * force_axial / ratio_nut
            else:
                Fn_sub = np.maximum(0, Fn[ii:number, :number_roller])
                sum_term = np.sum(Fn_sub) * ratio_nut * pitch_screw / (area_n * E_n)
                
                F_prev = np.maximum(0, Fn[ii-1, kk])
                F_curr = np.maximum(0, Fn[ii, kk])
                contact_term = (C_delta_inner * E0_inner**2 * sum_p_inner**(1/3) *
                               (F_prev**(2/3) - F_curr**(2/3)) / ratio_nut)
                
                flexible_term = flexible_inner * (F_prev - F_curr)
                modify_term = (modify_nut[ii-1, kk] - modify_nut[ii, kk])
                
                nut_sum = np.sum(np.maximum(0, Fn[ii:number, kk])) * ratio_nut
                screw_sum = np.sum(np.maximum(0, Fn[ii+number:2*number, kk])) * ratio_screw
                balance_term = (nut_sum - screw_sum) * pitch_screw / (area_r * E_r)
                
                FF[ii, kk] = unit_scale * (sum_term - contact_term - flexible_term - modify_term + balance_term)

        # 丝杠侧方程构建（关键差异部分）
        for ii in range(number):
            if ii == 0:
                FF[ii+number, kk] = np.sum(np.maximum(0, Fn[number:2*number, kk])) - share[kk, 0] * force_axial / ratio_screw
            else:
                # 差异点1：求和范围为从起始到当前前一行（原MATLAB代码中的1+number:ii-1+number）
                # Python切片为 [number : number+ii] 因为MATLAB的1+number到ii-1+number对应行数为number到number+ii-1
                Fn_sub = np.maximum(0, Fn[number : number+ii, :number_roller])  # 注意右边界不包含
                sum_term = np.sum(Fn_sub) * ratio_screw * pitch_screw / (area_s * E_s)
                
                # 差异点2：接触变形项符号相反（Fn当前行与前一行顺序对调）
                F_curr = np.maximum(0, Fn[number+ii, kk])
                F_prev = np.maximum(0, Fn[number+ii-1, kk])
                contact_term = (C_delta_outer * E0_outer**2 * sum_p_outer**(1/3) *
                              (F_curr**(2/3) - F_prev**(2/3)) / ratio_screw)  # 注意符号变化
                
                # 差异点3：柔度项符号相反
                flexible_term = flexible_outer * (F_curr - F_prev)  # 顺序对调
                
                # 差异点4：修形项符号相反
                modify_term = (modify_screw[ii, kk] - modify_screw[ii-1, kk])  # 项顺序对调
                
                # 滚柱平衡项（符号与原函数一致）
                nut_sum = np.sum(np.maximum(0, Fn[ii:number, kk])) * ratio_nut
                screw_sum = np.sum(np.maximum(0, Fn[number+ii : 2*number, kk])) * ratio_screw
                balance_term = (nut_sum - screw_sum) * pitch_screw / (area_r * E_r)
                
                # 组合所有项（注意balance_term符号）
                FF[ii+number, kk] = unit_scale * (sum_term - contact_term - flexible_term - modify_term + balance_term)
    
    return FF.flatten()