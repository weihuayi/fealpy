from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.cem.model.point_source_maxwell import PointSourceMaxwell

c0 = 299792458.0
mu0 = 4.0 * bm.pi * 1e-7
eta0: float = mu0 * c0

def test_convergence_3d():
    """3D 收敛性测试"""
    # 定义解析解函数
    time = 200
    m, n, p = 1, 1, 1
    E0 = 1.0
    c = 299_792_458.0
    mu0 = 4e-7 * bm.pi
    eps0 = 1.0 / (mu0 * c**2)

    # 波数和角频率
    kx = m * bm.pi
    ky = n * bm.pi
    kz = p * bm.pi
    kc2 = kx**2 + ky**2 + kz**2
    omega = c * bm.sqrt(kx**2 + ky**2 + kz**2)

    def Ex_func(X, Y, Z, t):
        return (
            -E0 * (kx * kz) / kc2
            * bm.cos(kx * X)
            * bm.sin(ky * Y)
            * bm.sin(kz * Z)
            * bm.cos(omega * t)
        )

    def Ey_func(X, Y, Z, t):
        return (
            -E0 * (ky * kz) / kc2
            * bm.sin(kx * X)
            * bm.cos(ky * Y)
            * bm.sin(kz * Z)
            * bm.cos(omega * t)
        )

    def Ez_func(X, Y, Z, t):
        return (
            E0
            * bm.sin(kx * X)
            * bm.sin(ky * Y)
            * bm.cos(kz * Z)
            * bm.cos(omega * t)
        )

    def Hx_func(X, Y, Z, t):
        return (
            -E0 * (omega * eps0 * ky) / kc2
            * bm.sin(kx * X)
            * bm.cos(ky * Y)
            * bm.cos(kz * Z)
            * bm.sin(omega * t)
        )

    def Hy_func(X, Y, Z, t):
        return (
            E0 * (omega * eps0 * kx) / kc2
            * bm.cos(kx * X)
            * bm.sin(ky * Y)
            * bm.cos(kz * Z)
            * bm.sin(omega * t)
        )

    def Hz_func(X, Y, Z, t):
        return (
            E0 * (omega * eps0) / kc2
            * bm.cos(kx * X)
            * bm.cos(ky * Y)
            * bm.sin(kz * Z)
            * bm.sin(omega * t)
        )

    h0 = 0.0625
    domain = [0, 1, 0, 1, 0, 1]
    time_steps = 200
    
    dt = bm.sqrt(2) * h0 / (c * 128)  # CFL条件
    
    L2_errors = []
    H1_errors = []
    maxit = 4  # 减少迭代次数以节省时间
    
    for i in range(maxit):
        print(f"开始第 {i+1} 次迭代，网格大小: {h0}")
        
        N = int((domain[1] - domain[0]) / h0)

        pde = PointSourceMaxwell(eps=1.0, mu=1.0, domain=domain)
        options = {
            'dt': dt,
            'boundary': 'PEC',
            'maxstep': time_steps,
            'save_every': time_steps
        }
        
        from fealpy.cem.point_source_maxwell_fdtd_model import PointSourceMaxwellFDTDModel
        model = PointSourceMaxwellFDTDModel(pde, N, options)
        
        # 初始化场
        model.E['x'] = model.mesh.initialize_field('E_x', Ex_func, dt)
        model.E['y'] = model.mesh.initialize_field('E_y', Ey_func, dt)
        model.E['z'] = model.mesh.initialize_field('E_z', Ez_func, dt)
        model.H['x'] = model.mesh.initialize_field('H_x', Hx_func, dt)
        model.H['y'] = model.mesh.initialize_field('H_y', Hy_func, dt)
        model.H['z'] = model.mesh.initialize_field('H_z', Hz_func, dt)
        
        # 运行模拟
        field_history = model.run(time_steps)
        
        # 获取模拟结果
        Ex_sim = model.E['x']
        Ey_sim = model.E['y']
        Ez_sim = model.E['z']
        Hx_sim = model.H['x']
        Hy_sim = model.H['y']
        Hz_sim = model.H['z']
        
        # 计算解析解
        t_final = time_steps * dt

        Ex_coords = model.mesh.edgex_coords
        Ey_coords = model.mesh.edgey_coords
        Ez_coords = model.mesh.edgez_coords
        
        # H场分量坐标
        Hx_coords = model.mesh.facex_coords
        Hy_coords = model.mesh.facey_coords
        Hz_coords = model.mesh.facez_coords
        
        # 计算E场解析解
        X_ex, Y_ex, Z_ex = Ex_coords[..., 0], Ex_coords[..., 1], Ex_coords[..., 2]
        Ex_exact = Ex_func(X_ex, Y_ex, Z_ex, t_final)
        
        X_ey, Y_ey, Z_ey = Ey_coords[..., 0], Ey_coords[..., 1], Ey_coords[..., 2]
        Ey_exact = Ey_func(X_ey, Y_ey, Z_ey, t_final)
        
        X_ez, Y_ez, Z_ez = Ez_coords[..., 0], Ez_coords[..., 1], Ez_coords[..., 2]
        Ez_exact = Ez_func(X_ez, Y_ez, Z_ez, t_final)
        
        # 计算H场解析解（半时间步）
        X_hx, Y_hx, Z_hx = Hx_coords[..., 0], Hx_coords[..., 1], Hx_coords[..., 2]
        Hx_exact = Hx_func(X_hx, Y_hx, Z_hx, t_final - 0.5 * dt)
        
        X_hy, Y_hy, Z_hy = Hy_coords[..., 0], Hy_coords[..., 1], Hy_coords[..., 2]
        Hy_exact = Hy_func(X_hy, Y_hy, Z_hy, t_final - 0.5 * dt)
        
        X_hz, Y_hz, Z_hz = Hz_coords[..., 0], Hz_coords[..., 1], Hz_coords[..., 2]
        Hz_exact = Hz_func(X_hz, Y_hz, Z_hz, t_final - 0.5 * dt)
        
        # 计算所有分量的误差
        error_Ex = Ex_sim - Ex_exact
        error_Ey = Ey_sim - Ey_exact
        error_Ez = Ez_sim - Ez_exact
        error_Hx = Hx_sim - Hx_exact
        error_Hy = Hy_sim - Hy_exact
        error_Hz = Hz_sim - Hz_exact
        
        # 计算每个分量的L2误差
        l2_error_Ex = bm.sqrt(bm.sum(error_Ex**2) * (h0**3))
        l2_error_Ey = bm.sqrt(bm.sum(error_Ey**2) * (h0**3))
        l2_error_Ez = bm.sqrt(bm.sum(error_Ez**2) * (h0**3))
        l2_error_Hx = bm.sqrt(bm.sum(error_Hx**2) * (h0**3))
        l2_error_Hy = bm.sqrt(bm.sum(error_Hy**2) * (h0**3))
        l2_error_Hz = bm.sqrt(bm.sum(error_Hz**2) * (h0**3))
        
        # 总L2误差
        l2_error = bm.sqrt(l2_error_Ex**2 + l2_error_Ey**2 + l2_error_Ez**2 + 
                          l2_error_Hx**2 + l2_error_Hy**2 + l2_error_Hz**2)
        
        # 计算H1误差（梯度误差）- 完整3D版本
        # 对每个场分量计算三个方向的梯度
        
        # Ex的梯度
        dx_Ex = error_Ex[1:, :, :] - error_Ex[:-1, :, :]
        dy_Ex = error_Ex[:, 1:, :] - error_Ex[:, :-1, :]
        dz_Ex = error_Ex[:, :, 1:] - error_Ex[:, :, :-1]
        
        # Ey的梯度
        dx_Ey = error_Ey[1:, :, :] - error_Ey[:-1, :, :]
        dy_Ey = error_Ey[:, 1:, :] - error_Ey[:, :-1, :]
        dz_Ey = error_Ey[:, :, 1:] - error_Ey[:, :, :-1]
        
        # Ez的梯度
        dx_Ez = error_Ez[1:, :, :] - error_Ez[:-1, :, :]
        dy_Ez = error_Ez[:, 1:, :] - error_Ez[:, :-1, :]
        dz_Ez = error_Ez[:, :, 1:] - error_Ez[:, :, :-1]
        
        # Hx的梯度
        dx_Hx = error_Hx[1:, :, :] - error_Hx[:-1, :, :]
        dy_Hx = error_Hx[:, 1:, :] - error_Hx[:, :-1, :]
        dz_Hx = error_Hx[:, :, 1:] - error_Hx[:, :, :-1]
        
        # Hy的梯度
        dx_Hy = error_Hy[1:, :, :] - error_Hy[:-1, :, :]
        dy_Hy = error_Hy[:, 1:, :] - error_Hy[:, :-1, :]
        dz_Hy = error_Hy[:, :, 1:] - error_Hy[:, :, :-1]
        
        # Hz的梯度
        dx_Hz = error_Hz[1:, :, :] - error_Hz[:-1, :, :]
        dy_Hz = error_Hz[:, 1:, :] - error_Hz[:, :-1, :]
        dz_Hz = error_Hz[:, :, 1:] - error_Hz[:, :, :-1]
        
        # 计算梯度项的平方和
        eu21 = (bm.sum(dx_Ex**2) + bm.sum(dy_Ex**2) + bm.sum(dz_Ex**2)) * h0
        eu22 = (bm.sum(dx_Ey**2) + bm.sum(dy_Ey**2) + bm.sum(dz_Ey**2)) * h0
        eu23 = (bm.sum(dx_Ez**2) + bm.sum(dy_Ez**2) + bm.sum(dz_Ez**2)) * h0
        eu24 = (bm.sum(dx_Hx**2) + bm.sum(dy_Hx**2) + bm.sum(dz_Hx**2)) * h0
        eu25 = (bm.sum(dx_Hy**2) + bm.sum(dy_Hy**2) + bm.sum(dz_Hy**2)) * h0
        eu26 = (bm.sum(dx_Hz**2) + bm.sum(dy_Hz**2) + bm.sum(dz_Hz**2)) * h0
        
        eu2 = eu21 + eu22 + eu23 + eu24 + eu25 + eu26
        
        # 计算L2部分的平方和
        eu11 = bm.sum(error_Ex**2) * (h0**3)
        eu12 = bm.sum(error_Ey**2) * (h0**3)
        eu13 = bm.sum(error_Ez**2) * (h0**3)
        eu14 = bm.sum(error_Hx**2) * (h0**3)
        eu15 = bm.sum(error_Hy**2) * (h0**3)
        eu16 = bm.sum(error_Hz**2) * (h0**3)
        eu1 = eu11 + eu12 + eu13 + eu14 + eu15 + eu16
        
        # H1误差 = sqrt(L2部分 + 梯度部分)
        h1_error = bm.sqrt(eu1 + eu2)
        
        # 存储误差
        L2_errors.append({"h": h0, "L2": l2_error})
        H1_errors.append({"h": h0, "H1": h1_error})
        
        # 计算收敛阶
        if i > 0:
            ln2 = bm.log(2)
            r1 = bm.log(L2_errors[i-1]["L2"] / l2_error) / ln2
            r2 = bm.log(H1_errors[i-1]["H1"] / h1_error) / ln2
            L2_errors[i]["rate"] = r1
            H1_errors[i]["rate"] = r2
            print(f"L2收敛阶: {r1:.4f}, H1收敛阶: {r2:.4f}")
        
        h0 = h0 / 2
        dt = dt / 2  # 同时减小时间步长以满足CFL条件
        del model, pde
    
    # 打印结果
    print("\n=== 3D L2 误差 ===")
    for i, err in enumerate(L2_errors):
        if i == 0:
            print(f"h={err['h']:.6f}, L2={err['L2']:.6e}")
        else:
            print(f"h={err['h']:.6f}, L2={err['L2']:.6e}, rate={err.get('rate', 'N/A'):.4f}")
    
    print("\n=== 3D H1 误差 ===")
    for i, err in enumerate(H1_errors):
        if i == 0:
            print(f"h={err['h']:.6f}, H1={err['H1']:.6e}")
        else:
            print(f"h={err['h']:.6f}, H1={err['H1']:.6e}, rate={err.get('rate', 'N/A'):.4f}")
    
    return L2_errors, H1_errors

def test_convergence_2d():
    """2D PEC边界收敛性测试"""
    # 定义解析解函数
    c = 299792458.0
    mu0 = 4e-7 * bm.pi
    m, n = 1, 1
    E0 = 1.0
    omega = c * bm.pi * bm.sqrt(bm.array((m)**2 + (n)**2))

    def Ez_func(X, Y, t):
        return E0 * bm.sin(m*bm.pi*X) * bm.sin(n*bm.pi*Y) * bm.cos(omega*t)

    def Hx_func(X, Y, t):
        return -E0/(mu0*omega) * (n*bm.pi) * bm.sin(m*bm.pi*X) * bm.cos(n*bm.pi*Y) * bm.sin(omega*t)

    def Hy_func(X, Y, t):
        return E0/(mu0*omega) * (m*bm.pi) * bm.cos(m*bm.pi*X) * bm.sin(n*bm.pi*Y) * bm.sin(omega*t)

    h0 = 0.0625
    domain = [0, 1, 0, 1]
    time_steps = 800
    
    dt = bm.sqrt(bm.array(2))*0.0625/(128*c)  
    
    L2_errors = []
    H1_errors = []
    maxit = 5
    
    for i in range(maxit):
        print(f"开始第 {i+1} 次迭代，网格大小: {h0}")
        
        N = int((domain[1] - domain[0]) / h0)

        pde = PointSourceMaxwell(eps=1.0, mu=1.0, domain=domain)
        options = {
            'dt': dt,
            'boundary': 'PEC',
            'maxstep': time_steps,
            'save_every': time_steps
        }
        
        from fealpy.cem.point_source_maxwell_fdtd_model import PointSourceMaxwellFDTDModel
        model = PointSourceMaxwellFDTDModel(pde, N, options)
        
        # 初始化场
        Ez = model.mesh.initialize_field('E_z', Ez_func, dt)
        Hx = model.mesh.initialize_field('H_x', Hx_func, dt)  
        Hy = model.mesh.initialize_field('H_y', Hy_func, dt)
        model.E['z'] = Ez 
        model.H['x'] = Hx
        model.H['y'] = Hy
        
        # 运行模拟
        field_history = model.run(time_steps)
        
        # 获取模拟结果
        Ez_sim = model.E['z'] 
        Hx_sim = model.H['x']
        Hy_sim = model.H['y']
        
        # 计算解析解
        t_final = (time_steps) * dt

        # 获取网格坐标
        node = model.mesh.node.reshape(N+1, N+1, 2).squeeze()
        X, Y = node[..., 0], node[..., 1]

        # E场在整时间步
        Ez_exact = Ez_func(X, Y, t_final)
        
        # H场在边中心，使用半时间步
        edgey_bc = model.mesh.edgey_coords  # 对应H_x
        edgex_bc = model.mesh.edgex_coords # 对应H_y
        X_hx, Y_hx = edgey_bc[..., 0], edgey_bc[..., 1]  # H_x坐标
        X_hy, Y_hy = edgex_bc[..., 0], edgex_bc[..., 1]  # H_y坐标
        
        # H场在半时间步（比E场晚半个时间步）
        Hx_exact = Hx_func(X_hx, Y_hx, t_final - 0.5 * dt)
        Hy_exact = Hy_func(X_hy, Y_hy, t_final - 0.5 * dt)
        
        # 计算误差
        eh1 = Ez_sim - Ez_exact
        eh2 = Hx_sim - Hx_exact  
        eh3 = Hy_sim - Hy_exact

        # L2误差
        eu11 = bm.sum(eh1**2) * (h0**2)
        eu12 = bm.sum(eh2**2) * (h0**2)
        eu13 = bm.sum(eh3**2) * (h0**2)
        eu1 = eu11 + eu12 + eu13
        l2_error = bm.sqrt(eu1)
        
        # H1误差（梯度项）
        dx1 = eh1[1:, :] - eh1[:-1, :]
        dy1 = eh1[:, 1:] - eh1[:, :-1]
        dx2 = eh2[1:, :] - eh2[:-1, :]
        dy2 = eh2[:, 1:] - eh2[:, :-1]
        dx3 = eh3[1:, :] - eh3[:-1, :]
        dy3 = eh3[:, 1:] - eh3[:, :-1]
        
        eu21 = (bm.sum(dx1**2) + bm.sum(dy1**2))
        eu22 = (bm.sum(dx2**2) + bm.sum(dy2**2))
        eu23 = (bm.sum(dx3**2) + bm.sum(dy3**2))
        eu2 = eu21 + eu22 + eu23
        
        h1_error = bm.sqrt(eu1 + eu2)
        
        # 存储误差
        L2_errors.append({"h": h0, "L2": l2_error})
        H1_errors.append({"h": h0, "H1": h1_error})
        
        # 计算收敛阶
        if i > 0:
            ln2 = bm.log(bm.array(2))
            r1 = bm.log(L2_errors[i-1]["L2"] / l2_error) / ln2
            r2 = bm.log(H1_errors[i-1]["H1"] / h1_error) / ln2
            L2_errors[i]["rate"] = r1
            H1_errors[i]["rate"] = r2
            print(f"L2收敛阶: {r1:.4f}, H1收敛阶: {r2:.4f}")
        
        h0 = h0 / 2
        del model, pde
    
    # 打印结果
    print("\n=== 2D L2 误差 ===")
    for i, err in enumerate(L2_errors):
        if i == 0:
            print(f"h={err['h']:.6f}, L2={err['L2']:.6e}")
        else:
            print(f"h={err['h']:.6f}, L2={err['L2']:.6e}, rate={err.get('rate', 'N/A'):.4f}")
    
    print("\n=== 2D H1 误差 ===")
    for i, err in enumerate(H1_errors):
        if i == 0:
            print(f"h={err['h']:.6f}, H1={err['H1']:.6e}")
        else:
            print(f"h={err['h']:.6f}, H1={err['H1']:.6e}, rate={err.get('rate', 'N/A'):.4f}")
    
    return L2_errors, H1_errors

if __name__ == "__main__":
    print("运行3D收敛性测试...")
    L2_3d, H1_3d = test_convergence_3d()
    
    print("\n" + "="*50 + "\n")
    
    print("运行2D PEC边界收敛性测试...")
    L2_2d, H1_2d = test_convergence_2d()