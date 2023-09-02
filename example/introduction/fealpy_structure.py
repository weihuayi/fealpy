import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_module(name, x, y, width, height, ax, color='lightblue'):
    """绘制一个模块"""
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, name, ha='center', va='center', fontsize=8)

def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    
    # 绘制主模块: FEALPy
    draw_module('FEALPy', 5, 6.5, 2, 1, ax, color='lightgreen')
    
    # 绘制子模块
    modules = ['mesh', 'functionspace', 'solver', 'fem', 'vem', 'wg', 'ml', 'opt']
    x_starts = [1, 3, 5, 7, 9, 4, 6, 8]
    y_positions = [4, 4, 4, 4, 4, 2, 2, 2]
    
    for i, module in enumerate(modules):
        draw_module(module, x_starts[i], y_positions[i], 2, 1, ax)
    
        # 连接到主模块
        if module in ['ml', 'opt', 'wg']:
            ax.annotate("", xy=(x_starts[i] + 1, y_positions[i] + 1), xytext=(6, 6), arrowprops=dict(arrowstyle="->"))
        else:
            ax.annotate("", xy=(x_starts[i] + 1, y_positions[i] + 1), xytext=(6, 6.5), arrowprops=dict(arrowstyle="->"))
    
    plt.axis('off')  # 隐藏坐标轴
    plt.title("FEALPy Modules", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


