from ..backend import bm
import matplotlib.pyplot as plt

class MeshQuality:
    def __init__(self, mesh,logic_mesh, M):
        self.mesh = mesh
        self.logic_mesh = logic_mesh
        self.M = M
        self.NC = mesh.number_of_cells()
        self.cell = mesh.entity('cell')
        self.d = mesh.GD
        self.cm = mesh.entity_measure('cell')
        self.area = bm.sum(self.cm)

    def edge_matrix(self,X):
        """
        边矩阵 E
        E = [x_1 - x_0, x_2 - x_0, ..., x_d - x_0]
        """
        cell = self.cell
        X0 = X[cell[:, 0], :]
        E = X[cell[:, 1:], :] - X0[:, None, :] # (NC, GD, GD)
        E = bm.permute_dims(E, axes=(0,2,1))
        return E
    
    def sigma(self):
        """
        计算 sigma 指标
        sigma = sum_K ( |K| * rho_K )
        """
        M = self.M
        rho = bm.sqrt(bm.linalg.det(M))
        cm = self.cm
        sigma = bm.sum(cm * rho , axis=0)
        return sigma
    
    def Jacobian(self):
        """
        计算雅可比矩阵 J_K
        J_K = E_K E_hat^{-1}
        """
        mesh = self.mesh
        logic_mesh = self.logic_mesh

        E = self.edge_matrix(mesh.entity('node'))
        E_hat = self.edge_matrix(logic_mesh.entity('node'))
        E_hat_inv = bm.linalg.inv(E_hat)
        J = bm.matmul(E, E_hat_inv)
        return J
    
    def Q_eq(self):
        """
        计算等分布指标 Q_eq
        Q_eq_K = |K| * det(M_K)^{1/2} / (sigma/area)
        Q_eq = ((1/NC) sum_K Q_eq_K**2)^{1/2}
        其中 NC 为单元总数
        """
        M = self.M
        NC = self.NC
        det_M = bm.linalg.det(M)
        
        cm = self.cm
        sigma = self.sigma()
        
        Q_eq_K = (cm * (det_M)**0.5) / (sigma/NC)
        self.mesh.celldata['Q_eq_K'] = 1/Q_eq_K
        Q_eq = bm.sqrt( bm.sum( Q_eq_K**2 ) / NC )
        return Q_eq
    
    def Q_ali(self):
        """
        计算对齐指标 Q_ali
        Q_ali_K = trace(J_K^T M_K J_K) / (d * det(J_K^T M_K J_K)^{1/d})
        Q_ali = ((1/NC) sum_K Q_ali_K**2)^{1/2}
        其中 NC 为单元总数
        """
        J = self.Jacobian()
        M = self.M
        d = self.d
        NC = self.NC
        JT = bm.permute_dims(J, axes=(0,2,1))
        A = JT@ M @ J
        trA = bm.trace(A , axis1=-2, axis2=-1)
        detA = bm.linalg.det(A)
        
        Q_ali_K = trA / ( d * (detA)**(1/d) )
        self.mesh.celldata['Q_ali_K'] = 1/Q_ali_K
        Q_ali = bm.sqrt( bm.sum( Q_ali_K**2 ) / NC )
        return Q_ali
    
    def Q_geo(self):
        """
        计算几何指标 Q_geo
        Q_geo_K = trace(J_K^T J_K) / (d * det(J_K^T J_K)^{1/d})
        Q_geo = ((1/NC) sum_K Q_geo_K**2)^{1/2}
        其中 NC 为单元总数
        """
        J = self.Jacobian()
        d = self.d
        NC = self.NC
        JT = bm.permute_dims(J, axes=(0,2,1))
        A = JT @ J
        trA = bm.trace(A , axis1=-2, axis2=-1)
        detA = bm.linalg.det(A)
        Q_geo_K = trA / ( d * (detA)**(1/d) )
        self.mesh.celldata['Q_geo_K'] = Q_geo_K
        Q_geo = bm.sqrt( bm.sum( Q_geo_K**2 ) / NC )
        return Q_geo
    
    def stats_hist_metrics(self, bins=50, logscale=False, savefig=None, percentiles=(1,5,25,50,75,95,99)):
        """
        统计并绘制逐单元质量指标的直方图，返回统计量字典。
        参数:
          bins: int 或序列，直方图箱数或边界
          logscale: bool, 是否对 x 轴使用对数刻度（当指标正且跨越量级时）
          savefig: None 或 文件路径，若非 None 则保存图像
          percentiles: 要计算的分位数列表
        返回:
          stats: dict, 包含 'Q_eq','Q_ali','Q_geo' 每项为统计信息字典
        """
        import numpy as _np

        # 取出单元指标（可能为 bm 张量或 numpy）
        def to_numpy(x):
            try:
                return _np.asarray(x)
            except Exception:
                try:
                    return _np.asarray(bm.to_numpy(x))
                except Exception:
                    return _np.asarray(x)
        all_Q_eq = self.Q_eq()
        all_Q_ali = self.Q_ali()
        all_Q_geo = self.Q_geo()
        
        Qeq = to_numpy(self.mesh.celldata.get('Q_eq_K'))
        Qali = to_numpy(self.mesh.celldata.get('Q_ali_K'))
        Qgeo = to_numpy(self.mesh.celldata.get('Q_geo_K'))
        metrics = {'Q_eq': Qeq.ravel(), 'Q_ali': Qali.ravel(), 'Q_geo': Qgeo.ravel()}
        global_map = {'Q_eq': all_Q_eq, 'Q_ali': all_Q_ali, 'Q_geo': all_Q_geo}
        
        stats = {}
        for name, arr in metrics.items():
            clean = arr[~_np.isnan(arr)]
            if clean.size == 0:
                st = {'count': 0, 'global': float(global_map[name])}
            else:
                st = {
                    'count': int(clean.size),
                    'min': float(_np.min(clean)),
                    'max': float(_np.max(clean)),
                    'mean': float(_np.mean(clean)),
                    'median': float(_np.median(clean)),
                    'std': float(_np.std(clean)),
                    'global': float(global_map[name])
                }
                perc = _np.percentile(clean, percentiles)
                for p, val in zip(percentiles, perc):
                    st[f'p{p}'] = float(val)
            stats[name] = st

        # 绘图
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
        fig, axs = plt.subplots(1, 3, figsize=(15, 4),dpi = 200, constrained_layout=True)
        colors = ['#4C72B0', '#55A868', '#C44E52']

        for ax, (name, arr), color in zip(axs, metrics.items(), colors):
            data = arr.ravel()
            data = data[~_np.isnan(data)]
            if data.size == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=10)
                ax.set_title(name, fontweight='bold')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue

            # 直方图
            if logscale:
                data_pos = data[data > 0]
                if data_pos.size == 0:
                    ax.text(0.5, 0.5, 'no positive data for logscale', ha='center', va='center', fontsize=10)
                    ax.set_title(name, fontweight='bold')
                    ax.set_xticks([]); ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    continue
                hist_data = data_pos
                ax.set_xscale('log')
                ax.set_xlabel('log-scale value')
            else:
                hist_data = data
                ax.set_xlabel('value')

            ax.hist(hist_data, bins=bins, color=color, edgecolor='#333333', linewidth=0.6, alpha=0.7)
            ax.set_title(name, fontweight='bold')
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
            ax.set_ylabel('count')

            # 统计线（均值/中位数/5%、95%分位）
            st = stats[name]
            if st['count'] > 0:
                mean_v = st['mean']; med_v = st['median']
                p5 = st.get('p5'); p95 = st.get('p95')
                for v, ls, lw, lbl, col in [
                    (mean_v, '-', 1.4, 'mean', '#2E86AB'),
                    (med_v, '--', 1.1, 'median', '#D95F02'),
                ]:
                    ax.axvline(v, color=col, linestyle=ls, linewidth=lw, alpha=0.9, label=lbl)
                if p5 is not None and p95 is not None:
                    ax.axvline(p5, color='#555555', linestyle=':', linewidth=1.0, alpha=0.9, label='p5/p95')
                    ax.axvline(p95, color='#555555', linestyle=':', linewidth=1.0, alpha=0.9)

                # 角标摘要
                txt = (f"n={st['count']}\nmax={st['max']:.3f}\nmin={st['min']:.3f}\n"
                       f"std={st['std']:.3g}\nglobal={st['global']:.3f}")
                ax.annotate(txt, xy=(0.97, 0.95), xycoords='axes fraction', ha='right', va='top',
                            fontsize=9, bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#666666', alpha=0.9))

            # 去掉顶部和右侧脊柱
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(frameon=False, loc='best')

        # 可选总标题（留空以便嵌入论文排版时自定义）
        # fig.suptitle('Mesh Quality Metrics', fontsize=14, fontweight='bold', y=1.02)

        if savefig:
            fig.savefig(savefig, dpi=400, bbox_inches='tight', transparent=True)
        plt.show()

        return stats