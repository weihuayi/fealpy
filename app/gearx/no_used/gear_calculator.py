from fealpy.backend import backend_manager as bm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import json


def involute(mn, z, alpha_n, beta, x, t):
    """
    齿面渐开线函数
    :param mn: 法向模数
    :param z: 齿数
    :param alpha_n: 法向压力角
    :param beta: 螺旋角
    :param x: 变位系数
    :param t: 参变量
    :return: 参变量对应的渐开线点坐标
    """
    mt = mn / bm.cos(beta)  # 端面模数
    r = 0.5 * mt * z  # 分度圆半径
    k = -(bm.pi * mn / 4 + mn * x * bm.tan(alpha_n))
    phi = (t * bm.cos(bm.pi / 2 - alpha_n) ** 2 + k * bm.cos(bm.pi / 2 - alpha_n) + t * bm.cos(beta) ** 2 * bm.sin(
        bm.pi / 2 - alpha_n) ** 2) / (r * bm.cos(beta) * bm.cos(bm.pi / 2 - alpha_n))

    xt = (r * bm.sin(phi) - phi * r * bm.cos(phi) +
          t * bm.sin(phi) * bm.sin(bm.pi / 2 - alpha_n) +
          (bm.cos(phi) * (k + t * bm.cos(bm.pi / 2 - alpha_n))) / bm.cos(beta))

    yt = (r * bm.cos(phi) + phi * r * bm.sin(phi) +
          t * bm.cos(phi) * bm.sin(bm.pi / 2 - alpha_n) -
          (bm.sin(phi) * (k + t * bm.cos(bm.pi / 2 - alpha_n))) / bm.cos(beta))

    return xt, yt


def involutecross_func(mn, z, alpha_n, beta, x, t):
    """
    齿面渐开线函数，用于求解渐开线与齿顶圆的交点
    :param mn: 法向模数
    :param z: 齿数
    :param alpha_n: 法向压力角
    :param beta: 螺旋角
    :param x: 变位系数
    :param t: 参变量
    :return: 参变量对应的渐开线点极径
    """
    xt, yt = involute(mn, z, alpha_n, beta, x, t)
    return bm.sqrt(xt ** 2 + yt ** 2)


def transition(mn, z, alpha_n, beta, hac, cc, rco, x, t):
    """
    齿根过渡曲线渐开线函数
    :param mn: 法向模数
    :param z: 齿数
    :param alpha_n: 法向压力角
    :param beta: 螺旋角
    :param hac: 齿顶高系数
    :param cc: 顶隙系数
    :param rco: 刀尖圆弧半径
    :param x: 刀尖圆弧坐标点
    :param t: 参变量
    :return:
    """
    mt = mn / bm.cos(beta)  # 端面模数
    r = 0.5 * mt * z  # 分度圆半径
    r0 = mn * rco  # 刀尖圆弧半径
    cha = (hac + cc) * mn  # 刀具齿顶高

    # 刀尖圆弧 y 坐标
    x0 = -bm.pi * mn / 2 + (bm.pi * mn / 4 - cha * bm.tan(alpha_n) - r0 * bm.tan(0.25 * bm.pi - 0.5 * alpha_n))
    # 刀尖圆弧 y 坐标
    y0 = -(cha - r0) + mn * x

    phi = (x0 * bm.sin(t) + r0 * bm.cos(t) * bm.sin(t) - y0 * bm.cos(beta) ** 2 * bm.cos(t) - r0 * bm.cos(
        beta) ** 2 * bm.cos(t) * bm.sin(t)) / (r * bm.cos(beta) * bm.sin(t))

    xt = (r * bm.sin(phi) + bm.sin(phi) * (y0 + r0 * bm.sin(t)) - phi * r * bm.cos(phi) +
          (bm.cos(phi) * (x0 + r0 * bm.cos(t))) / bm.cos(beta))

    yt = (r * bm.cos(phi) + bm.cos(phi) * (y0 + r0 * bm.sin(t)) + phi * r * bm.sin(phi) -
          (bm.sin(phi) * (x0 + r0 * bm.cos(t))) / bm.cos(beta))

    return xt, yt


def produce_involute_gear_profile_points(mn, z, alpha_n, beta, x, hac, cc, rco, involute_section, transition_section):
    xt1 = bm.zeros((involute_section + transition_section + 1) * 2)
    yt1 = bm.zeros((involute_section + transition_section + 1) * 2)
    points = bm.zeros(((involute_section + transition_section + 1) * 2, 3))
    mt = mn / bm.cos(beta)  # 端面模数

    # s1 = 0.5 * jn / cos(alpha_n)  # 法向侧隙产生的齿厚减薄量
    # x = x - s1 / (2 * mn * tan(alpha_n))  # 考虑法向侧隙的实际加工使用的变位系数

    d = mt * z  # 分度圆直径
    da = d + 2 * mn * (hac + x)  # 齿顶圆直径
    cha = (hac + cc) * mn  # 刀具齿顶高
    r0 = mn * rco  # 刀尖圆弧半径

    t1 = (mn * x - (cha - r0 + r0 * bm.sin(alpha_n))) / bm.cos(alpha_n)

    def involutecross(t2):
        return involutecross_func(mn, z, alpha_n, beta, x, t2) - (0.5 * da)

    t2 = fsolve(involutecross, mn)[0]  # 求解渐开线与齿顶圆的交点

    t3 = 2 * bm.pi - alpha_n
    t4 = 1.5 * bm.pi
    width2 = t3 - t4
    t = t4 - width2 / transition_section

    for i in range(transition_section + 1):
        t += width2 / transition_section
        xt1[i], yt1[i] = transition(mn, z, alpha_n, beta, hac, cc, rco, x, t)

    width1 = t2 - t1
    t = t1 - width1 / involute_section

    for i in range(transition_section + 1, involute_section + transition_section + 1):
        t += width1 / involute_section
        xt1[i], yt1[i] = involute(mn, z, alpha_n, beta, x, t)

    for i in range(involute_section + transition_section + 1):
        xt1[involute_section + transition_section + 1 + i] = -xt1[i]
        yt1[involute_section + transition_section + 1 + i] = yt1[i]

    for i in range((involute_section + transition_section + 1) * 2):
        points[i, 0] = xt1[i]
        points[i, 1] = yt1[i]
        points[i, 2] = 0

    return points


if __name__ == '__main__':
    # 参数读取
    with open('cal_gear_data.json', 'r') as file:
        data = json.load(file)
    mn = data['mn']  # 法向模数
    z = data['z']  # 齿数
    alpha_n = data['alpha_n']  # 法向压力角
    beta = data['beta']  # 螺旋角
    x = data['x']  # 变位系数
    hac = data['hac']  # 齿顶高系数
    cc = data['cc']  # 顶隙系数
    rco = data['rco']  # 刀尖圆弧半径
    # jn = data['jn']  # 法向侧隙
    involute_section = data['involute_section']  # 渐开线分段数
    transition_section = data['transition_section']  # 过渡曲线分段数

    alpha_n = alpha_n/180*bm.pi
    beta = beta/180*bm.pi
    mt = mn / bm.cos(beta)  # 端面模数

    d = mt * z  # 分度圆直径
    da = d + 2 * mn * (hac + x)  # 齿顶圆直径
    nn = involute_section+transition_section+1

    points = produce_involute_gear_profile_points(mn, z, alpha_n, beta, x, hac, cc, rco, involute_section,
                                                  transition_section)



    rot_phi = bm.linspace(0, 2 * bm.pi, z, endpoint=False)
    phi = bm.linspace(0, 2 * bm.pi, nn * 10)

    fig, ax = plt.subplots()
    # 绘制分度圆
    x_p = 0.5 * d * bm.cos(phi)
    y_p = 0.5 * d * bm.sin(phi)
    ax.plot(x_p, y_p)
    # 绘制齿顶圆
    x_a = 0.5*da*bm.cos(phi)
    y_a = 0.5*da*bm.sin(phi)
    ax.plot(x_a, y_a)
    # 绘制齿轮
    for i in range(z):
        theta = rot_phi[i]
        x_new = bm.cos(theta)*points[:, 0]-bm.sin(theta)*points[:, 1]
        y_new = bm.sin(theta)*points[:, 0]+bm.cos(theta)*points[:, 1]
        ax.plot(x_new[:nn], y_new[:nn])
        ax.plot(x_new[nn:], y_new[nn:])
    plt.axis('equal')
    plt.show()
