"""
扩展实验绘图（论文用，英文标注）：读取 exp/results_new_E{4,5}.csv
  1) exp/figures/fig_e4_ext_noise_mse.png —— E4 扩展噪声/极端场景 MSE 柱状图（log 轴）
  2) exp/figures/fig_e5_extreme_imbalance_f1.png —— E5 极端失衡 Macro-F1 柱状图

用法：python experiments/plot_extension_results.py
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp')
FIG_DIR = os.path.join(EXP_DIR, 'figures')
# 论文项目 images/ 目录：所有图同步输出 PDF+PNG，供 LaTeX 正文引用
IMG_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'ICML26_Workshop_BIAM', 'images')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# BIAM2 用深红、BIAM 用蓝：与 plot_biam2_analysis.py 的配色约定保持一致
COLORS = {'BIAM2': '#c0392b', 'BIAM': '#2c6fbb', 'BIAM-B': '#7fb3d5', 'Lasso': '#1f77b4',
          'SAM': '#2ca02c', 'Huber': '#17becf', 'RANSAC': '#bcbd22', 'NAM': '#9467bd',
          'MLP': '#ff7f0e', 'XGBoost': '#8c564b', 'XGB-BAL': '#e377c2'}


def save_fig(fig, fname):
    """同时保存：exp/figures PNG + 论文 images/ PDF+PNG"""
    base = os.path.splitext(fname)[0]
    fig.savefig(os.path.join(FIG_DIR, base + '.png'), bbox_inches='tight')
    if IMG_DIR:
        os.makedirs(IMG_DIR, exist_ok=True)
        fig.savefig(os.path.join(IMG_DIR, base + '.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(IMG_DIR, base + '.png'), bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {base}.png / {base}.pdf')


def bar_matrix(df, block, metric, scenarios, fname, ylabel, methods, logy=False,
               rotate=20, width=0.11, figsize=None):
    methods = [m for m in methods if m in df.method.unique()]
    n_sc, n_me = len(scenarios), len(methods)
    figsize = figsize or (1.35 * n_sc + 1.5, 4.6)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_sc)
    for i, m in enumerate(methods):
        means, stds = [], []
        for sc in scenarios:
            sel = df[(df.block == block) & (df.scenario == sc) & (df.method == m)][metric]
            means.append(sel.mean() if len(sel) else np.nan)
            stds.append(sel.std() if len(sel) > 1 else 0.0)
        off = (i - (n_me - 1) / 2) * width
        ax.bar(x + off, means, width * 0.92, yerr=stds, capsize=2,
               label=m, color=COLORS.get(m), edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=rotate, ha='right')
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(ncol=min(len(methods), 5), fontsize=8)
    fig.tight_layout()
    save_fig(fig, fname)


def main():
    # ---- E4 ----
    # 论文收录范围：E4 是加性模型的极端机制压力测试（BIAM 为主角），
    # BIAM2 仅进 E1 腐蚀族与 §7 交互基准（双档调度），此处过滤
    p4 = os.path.join(EXP_DIR, 'results_new_E4.csv')
    if os.path.exists(p4):
        e4 = pd.read_csv(p4)
        e4 = e4[e4.method != 'BIAM2']
        if len(e4):
            sc = ['gauss', 'laplace', 'cauchy', 'lognormal', 'outlier50', 'outlier70',
                  'smalln', 'highdim']
            sc = [s for s in sc if s in e4.scenario.unique()]
            reg_methods = ['BIAM', 'BIAM-B', 'Lasso', 'SAM', 'Huber', 'RANSAC',
                           'NAM', 'MLP', 'XGBoost']
            # 命名与论文 images/ 引用一致（synthetic_extreme_mse）
            bar_matrix(e4, 'E4', 'mse', sc, 'synthetic_extreme_mse.png',
                       'Test MSE (log)', reg_methods, logy=True, rotate=25, width=0.09)
            bar_matrix(e4, 'E4', 'mae', sc, 'fig_e4_ext_noise_mae.png',
                       'Test MAE (log)', reg_methods, logy=True, rotate=25, width=0.09)

    # ---- E5 ----
    # 同 E4：BIAM2 不进 E5 图（极端失衡场景主角是 BIAM）
    p5 = os.path.join(EXP_DIR, 'results_new_E5.csv')
    if os.path.exists(p5):
        e5 = pd.read_csv(p5)
        e5 = e5[e5.method != 'BIAM2']
        if len(e5):
            sc = [s for s, *_ in
                  [('imb20_clean',), ('imb50_clean',), ('n20_imb20',),
                   ('n30_imb50',), ('n20_imb50_miss20',)] if s in e5.scenario.unique()]
            clf_methods = ['BIAM', 'BIAM-B', 'Lasso', 'SAM', 'NAM', 'MLP',
                           'XGBoost', 'XGB-BAL']
            bar_matrix(e5, 'E5', 'f1_macro', sc, 'fig_e5_extreme_imbalance_f1.png',
                       'Test Macro-F1', clf_methods, rotate=20, width=0.09,
                       figsize=(1.5 * len(sc) + 1.5, 4.6))
            bar_matrix(e5, 'E5', 'accuracy', sc, 'fig_e5_extreme_imbalance_acc.png',
                       'Test Accuracy', clf_methods, rotate=20, width=0.09,
                       figsize=(1.5 * len(sc) + 1.5, 4.6))


if __name__ == '__main__':
    main()
