import json
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    # 1. 读取数据
    file_path = 'data/isoflops_curves.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'.")
        return

    # 2. 分组并获取经验最优点 (N_opt)
    budget_groups = defaultdict(list)
    for run in data:
        budget_groups[run['compute_budget']].append(run)

    c_values, n_opt_values = [], []
    for c, runs in budget_groups.items():
        best_run = min(runs, key=lambda x: x['final_loss'])
        c_values.append(c)
        n_opt_values.append(best_run['parameters'])

    c_values = np.array(c_values)
    n_opt_values = np.array(n_opt_values)
    
    sort_indices = np.argsort(c_values)
    c_values = c_values[sort_indices]
    n_opt_values = n_opt_values[sort_indices]

    # 根据 C ≈ 6ND 计算对应的最佳数据集大小 D_opt
    d_opt_values = c_values / (6.0 * n_opt_values)

    # 3. 在对数空间进行幂律拟合 (N_opt = k * C^a)
    def log_power_law(log_c, log_k, a):
        return log_k + a * log_c

    # 拟合参数量 N
    popt_n, _ = curve_fit(log_power_law, np.log10(c_values), np.log10(n_opt_values))
    k_n = 10 ** popt_n[0]
    a_n = popt_n[1]

    # 拟合数据量 D
    popt_d, _ = curve_fit(log_power_law, np.log10(c_values), np.log10(d_opt_values))
    k_d = 10 ** popt_d[0]
    a_d = popt_d[1]

    # 4. 推测 10^23 和 10^24 FLOPs 下的数值
    budgets_to_predict = np.array([1e23, 1e24])
    n_preds = k_n * (budgets_to_predict ** a_n)
    d_preds = k_d * (budgets_to_predict ** a_d)

    # 打印 Deliverables 要求的句子
    print("Deliverable (a):")
    print(f"For compute budgets of 10^23 and 10^24 FLOPs, the predicted optimal model sizes are approximately {n_preds[0]/1e9:.1f} Billion and {n_preds[1]/1e9:.1f} Billion parameters, respectively.")
    
    print("\nDeliverable (b):")
    print(f"For compute budgets of 10^23 and 10^24 FLOPs, the predicted optimal dataset sizes are approximately {d_preds[0]/1e9:.1f} Billion and {d_preds[1]/1e9:.1f} Billion tokens, respectively.")

    # 5. 可视化图表
    c_extrapolate = np.logspace(18, 24.5, 100) # 外推至超过 10^24
    n_extrapolate = k_n * (c_extrapolate ** a_n)
    d_extrapolate = k_d * (c_extrapolate ** a_d)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 图 (a): 参数量 N_opt 缩放定律
    ax1.scatter(c_values, n_opt_values, color='blue', label='Empirical $N_{opt}$ (from IsoFLOPs)', zorder=3)
    ax1.plot(c_extrapolate, n_extrapolate, 'k--', label=f'Fit: $N_{{opt}} = {k_n:.2f} \\times C^{{{a_n:.4f}}}$', zorder=2)
    ax1.scatter(budgets_to_predict, n_preds, color='red', marker='*', s=150, label='Predictions ($10^{23}, 10^{24}$)', zorder=4)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Compute Budget $C$ (FLOPs)')
    ax1.set_ylabel('Optimal Model Size $N_{opt}$ (Parameters)')
    ax1.set_title('(a) Extrapolated Compute-Optimal Model Size')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    # 图 (b): 数据量 D_opt 缩放定律
    ax2.scatter(c_values, d_opt_values, color='green', label='Empirical $D_{opt}$ (calculated from $C/6N$)', zorder=3)
    ax2.plot(c_extrapolate, d_extrapolate, 'k--', label=f'Fit: $D_{{opt}} = {k_d:.4f} \\times C^{{{a_d:.4f}}}$', zorder=2)
    ax2.scatter(budgets_to_predict, d_preds, color='red', marker='*', s=150, label='Predictions ($10^{23}, 10^{24}$)', zorder=4)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Compute Budget $C$ (FLOPs)')
    ax2.set_ylabel('Optimal Dataset Size $D_{opt}$ (Tokens)')
    ax2.set_title('(b) Extrapolated Compute-Optimal Dataset Size')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

"""
printed:
Empirical Minimums (Lowest Loss per Compute Budget):
-----------------------------------------------------------------
Compute: 6.00e+18 | Optimal Params: 7.62e+08 | Loss: 5.8999
Compute: 1.00e+19 | Optimal Params: 8.07e+08 | Loss: 5.6179
Compute: 3.00e+19 | Optimal Params: 1.54e+09 | Loss: 5.1072
Compute: 6.00e+19 | Optimal Params: 1.95e+09 | Loss: 4.8306
Compute: 1.00e+20 | Optimal Params: 3.25e+09 | Loss: 4.6529
Compute: 3.00e+20 | Optimal Params: 5.90e+09 | Loss: 4.3112
Compute: 6.00e+20 | Optimal Params: 6.97e+09 | Loss: 4.1212
Compute: 1.00e+21 | Optimal Params: 6.86e+09 | Loss: 4.0028
Compute: 3.00e+21 | Optimal Params: 1.21e+10 | Loss: 3.7732
-----------------------------------------------------------------

Fitted Parameter Scaling Law:
Equation: N_opt(C) = k * C^a
k (coefficient) = 1.1634e+00
a (exponent)    = 0.4687
"""