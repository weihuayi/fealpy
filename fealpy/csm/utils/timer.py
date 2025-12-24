from typing import List, Generator, Tuple, Dict, Optional
from time import time
import sys

# 尝试导入 matplotlib，失败则禁用绘图
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _timer_core() -> Generator[None, str, List[Tuple[Optional[str], float]]]:
    tag_list: List[Optional[str]] = [None]
    time_list: List[float] = [time()]

    while True:
        tag = yield
        if tag is None:
            break
        tag_list.append(tag)
        time_list.append(time())

    return list(zip(tag_list, time_list))


def timer(draw_pie_chart: bool = False) -> Generator[None, str, None]:
    """
    A generator timer with optional pie chart visualization of time proportions.

    Parameters:
        draw_pie_chart (bool): If True and matplotlib is available, show a pie chart.
    """
    while True:
        result = yield from _timer_core()
        print(f"Timer received None and paused.")
        print(
"=================================================\n"
"   ID       Time        Proportion(%)    Label\n"
"-------------------------------------------------"
        )
        prev_time = result[0][1]
        total_time = result[-1][1] - prev_time

        events: List[Tuple[str, float]] = []
        for i in range(1, len(result)):
            label = result[i][0]
            curr_time = result[i][1]
            delta = curr_time - prev_time
            if label is not None:
                events.append((label, delta))
            prev_time = curr_time

        if total_time <= 0:
            print("  No valid time recorded.")
            print("=================================================")
            continue

        # Aggregate by first occurrence
        label_total: Dict[str, float] = {}
        label_first_index: Dict[str, int] = {}
        for idx, (label, delta) in enumerate(events):
            if label not in label_total:
                label_total[label] = 0.0
                label_first_index[label] = idx
            label_total[label] += delta

        sorted_labels = sorted(label_total.keys(), key=lambda lbl: label_first_index[lbl])

        # Prepare data for output and plotting
        labels_for_plot = []
        proportions_for_plot = []

        for i, label in enumerate(sorted_labels, start=1):
            delta = label_total[label]
            p = delta / total_time * 100

            # Store for pie chart
            labels_for_plot.append(label)
            proportions_for_plot.append(p)

            # Console output (unchanged)
            i_text = f"{i}".rjust(3)
            if delta > 1.0:
                time_text = f"{delta:.3f}".rjust(7) + " [s] "
            elif delta > 0.001:
                time_text = f"{delta*1e3:.3f}".rjust(7) + " [ms]"
            else:
                time_text = f"{delta*1e6:.3f}".rjust(7) + " [us]"
            p_text = f"{p:.3f}".rjust(12)
            print("  " + "    ".join([i_text, time_text, p_text, label]))

        # ===== 新增：绘制饼图 =====
        if draw_pie_chart and MATPLOTLIB_AVAILABLE and total_time > 0:
            try:
                # === 新增：支持中文显示 ===
                plt.rcParams['font.sans-serif'] = [
                    'SimHei',          # Windows 黑体
                    'PingFang HK',     # macOS
                    'WenQuanYi Micro Hei',  # Linux
                    'Microsoft YaHei',      # 微软雅黑
                    'DejaVu Sans'           # fallback
                ]
                plt.rcParams['axes.unicode_minus'] = False
                # ========================

                plt.figure(figsize=(8, 6))
                def autopct_format(pct):
                    return f'{pct:.1f}%' if pct > 0.5 else ''

                wedges, texts, autotexts = plt.pie(
                    proportions_for_plot,
                    labels=labels_for_plot,
                    autopct=autopct_format,
                    startangle=90,
                    textprops={'fontsize': 9}
                )
                plt.title("时间占比饼状图", fontsize=14, pad=20)  # 中文标题
                plt.axis('equal')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Warning: Failed to draw pie chart: {e}", file=sys.stderr)

        print("=================================================")