import numpy as np
import matplotlib.pyplot as plt

def plot_fitted_scaling_law():
    """
    使用 IsoFLOPs 方法拟合出的具体参数来绘制 LLM 缩放曲线 N_opt = k * C^a。
    展示外推至未来算力预算（10^23 和 10^24 FLOPs）。
    """
    # 1. 使用之前从 IsoFLOPs 数据拟合得到的具体 Scaling Law 参数
    # N_opt(C) = k * C^a
    k_val = 1.1634   # 拟合得出的系数 k
    alpha_val = 0.4687 # 拟合得出的指数 a

    # 2. 生成用于绘图的数据点，外推至超过 10^24 FLOPs
    c_draw = np.logspace(18, 25.5, num=200)

    # 3. 计算对应的 N_opt 值
    n_opt_draw = k_val * (c_draw ** alpha_val)

    # 4. 创建图形和坐标轴
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()

    # 使用双对数坐标绘制拟合的 Scaling Law 曲线
    ax.loglog(c_draw, n_opt_draw, linestyle='--', color='k', linewidth=2, 
              label=f'Fitted Scaling Law: $N_{{opt}} = {k_val:.4f} \\times C^{{{alpha_val:.4f}}}$')

    # 5. 绘制特定的外推点 (从上一轮计算结果)
    budgets_to_predict = np.array([1e23, 1e24])
    n_preds = k_val * (budgets_to_predict ** alpha_val)
    
    ax.scatter(budgets_to_predict, n_preds, color='red', marker='*', s=150, 
               label='Predictions at $10^{23}, 10^{24}$ FLOPs', zorder=5)

    # 在特定点附近添加文本标注
    for c, n in zip(budgets_to_predict, n_preds):
        ax.text(c * 1.5, n, f'{n/1e9:.1f}B Parameters', color='red', fontsize=10, fontweight='bold', va='center')

    # 6. 添加坐标轴标签和标题
    ax.set_xlabel('Compute Budget $C$ (FLOPs)')
    ax.set_ylabel('Optimal Model Size $N_{opt}$ (Parameters)')
    ax.set_title(f'LLM Optimal Parameter Scaling $N = k \\cdot C^\\alpha$ (Based on Example Fit)')

    # 7. 添加网格和图例
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(loc='lower right')

    # 调整布局以适应文本标注
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_fitted_scaling_law()