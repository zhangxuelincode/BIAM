"""
汇总与绘图：读取 exp/results_new_E{1,2,3}.csv，生成
  1) exp/results_new_summary.csv  —— mean±std 汇总表
  2) exp/figures/*.png            —— 对比实验图（报告引用）

用法：python experiments/plot_new_results.py
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

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

METHOD_ORDER = ['BIAM2', 'BIAM', 'BIAM-B', 'BIAM-I', 'Lasso', 'SAM', 'NAM', 'MLP', 'XGBoost']
# BIAM2 用深红、BIAM 用蓝：与 plot_biam2_analysis.py 的配色约定保持一致
COLORS = {'BIAM2': '#c0392b', 'BIAM': '#2c6fbb', 'BIAM-B': '#7fb3d5', 'BIAM-I': '#e377c2',
          'Lasso': '#1f77b4', 'SAM': '#2ca02c', 'NAM': '#9467bd',
          'MLP': '#ff7f0e', 'XGBoost': '#8c564b'}


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


def load_blocks():
    dfs = []
    for b in ('E1', 'E2', 'E3'):
        path = os.path.join(EXP_DIR, f'results_new_{b}.csv')
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        raise SystemExit('没有找到结果 CSV')
    return pd.concat(dfs, ignore_index=True)


def agg(df, metric):
    g = df.groupby(['block', 'scenario', 'method'])[metric].agg(['mean', 'std', 'count']).reset_index()
    return g


def fmt_mean_std(df, block, scenario, method, metric):
    sel = df[(df.block == block) & (df.scenario == scenario) & (df.method == method)][metric]
    if len(sel) == 0:
        return '-'
    if len(sel) == 1:
        return f'{sel.iloc[0]:.3f}'
    return f'{sel.mean():.3f}±{sel.std():.3f}'


def bar_matrix(df, block, metric, scenarios, fname, ylabel, logy=False,
               methods=None, rotate=20, width=0.13, figsize=None):
    methods = methods or [m for m in METHOD_ORDER if m in df.method.unique()]
    n_sc, n_me = len(scenarios), len(methods)
    figsize = figsize or (1.6 * n_sc + 1.5, 4.6)
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
    ax.legend(ncol=min(len(methods), 4), fontsize=8)
    fig.tight_layout()
    save_fig(fig, fname)


def line_plot(df, block, metric, x_map, fname, xlabel, ylabel, hue='method',
              subplot_col=None, methods=None, logy=False):
    methods = methods or [m for m in METHOD_ORDER if m in df.method.unique()]
    subplots = sorted(df[subplot_col].unique()) if subplot_col else [None]
    fig, axes = plt.subplots(1, len(subplots), figsize=(5.2 * len(subplots), 4.4),
                             sharey=True, squeeze=False)
    for ax, sp in zip(axes[0], subplots):
        d = df[df[subplot_col] == sp] if subplot_col else df
        for m in methods:
            dm = d[d.method == m]
            if not len(dm):
                continue
            g = dm.groupby('scenario')[metric].agg(['mean', 'std']).reset_index()
            g['x'] = g.scenario.map(x_map)
            g = g.sort_values('x')
            ax.plot(g.x, g['mean'], marker='o', ms=4, label=m, color=COLORS.get(m))
            ax.fill_between(g.x, g['mean'] - g['std'], g['mean'] + g['std'],
                            alpha=0.12, color=COLORS.get(m))
        ax.set_xlabel(xlabel)
        ax.set_title(f'{subplot_col}={sp}' if subplot_col else block)
        ax.grid(alpha=0.3, linewidth=0.5)
        if logy:
            ax.set_yscale('log')
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, fname)


def main():
    df = load_blocks()

    # ---- 汇总表 ----
    rows = []
    for block, metric, extra in (('E1', 'mse', ['mae']), ('E2', 'mse', ['mae']),
                                 ('E3', 'accuracy', ['f1_macro'])):
        sub = df[df.block == block]
        for sc in sub.scenario.unique():
            for m in sub.method.unique():
                sel = sub[(sub.scenario == sc) & (sub.method == m)]
                if not len(sel):
                    continue
                row = {'block': block, 'scenario': sc, 'method': m, 'n_seeds': len(sel)}
                for met in [metric] + extra:
                    mu, sd = sel[met].mean(), sel[met].std() if len(sel) > 1 else 0
                    row[f'{met}_mean'] = round(mu, 4)
                    row[f'{met}_std'] = round(sd, 4)
                    row[f'{met}_mean±std'] = f'{mu:.4f}±{sd:.4f}'
                if 'time_sec' in sel:
                    row['time_sec_mean'] = round(sel['time_sec'].mean(), 1)
                rows.append(row)
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(EXP_DIR, 'results_new_summary.csv')
    summary.to_csv(out_csv, index=False)
    print(f'[csv] {out_csv}')

    # ---- E1 图：噪声谱系 MSE（log 轴）----
    # 命名与论文 images/ 引用一致（synthetic_noise_mse）
    e1 = df[df.block == 'E1']
    sc_order = ['clean', 'mixture_A', 'mixture_B', 'heavy_t', 'outlier10', 'outlier30']
    sc_order = [s for s in sc_order if s in e1.scenario.unique()]
    bar_matrix(e1, 'E1', 'mse', sc_order, 'synthetic_noise_mse.png',
               'Test MSE (log)', logy=True, rotate=25, width=0.11)

    # ---- E2 图：缺失比例扫描（MCAR）+ 机制对比 ----
    # 论文收录范围：BIAM2 仅进 E1 腐蚀族与 §7 交互基准（双档调度），
    # E2 的主角是 BIAM/BIAM-I，图中不画 BIAM2 线
    e2 = df[(df.block == 'E2') & (df.method != 'BIAM2')].copy()
    e2['ratio'] = e2.scenario.str.extract(r'_(\d+)').astype(float) / 100
    e2['mech'] = e2.scenario.str.split('_').str[0]
    mcar = e2[e2.mech == 'MCAR']
    if len(mcar):
        line_plot(mcar, 'E2', 'mse', lambda s: float(s.split('_')[1]) / 100,
                  'fig_e2_mcar_ratio_mse.png', '缺失比例 (MCAR)', 'Test MSE')
    mechs = [m for m in ['MCAR', 'MAR', 'MNAR'] if m in e2.mech.unique()]
    e2_r2 = e2[e2.ratio == 0.2]
    if len(e2_r2):
        e2_r2 = e2_r2.copy()
        e2_r2['scenario'] = e2_r2['mech']
        bar_matrix(e2_r2, 'E2', 'mse', mechs, 'fig_e2_mechanism_mse.png',
                   'Test MSE', rotate=0, figsize=(7.5, 4.6), width=0.10)

    # ---- E3 图：ACC vs 标签噪声比例（按 r2 分面）----
    # 同 E2：BIAM2 不进 E3 图（失衡/噪声场景主角是 BIAM/BIAM-B）
    e3 = df[(df.block == 'E3') & (df.method != 'BIAM2')].copy()
    e3 = e3[e3.scenario != 'clean']
    e3['r1'] = e3.scenario.str.extract(r'r1=([\d.]+)').astype(float)
    e3['r2'] = e3.scenario.str.extract(r'r2=([\d.]+)').astype(float)
    if len(e3):
        x_map = dict(zip(e3.scenario, e3.r1))
        line_plot(e3, 'E3', 'accuracy', x_map,
                  'fig_e3_acc_vs_noise.png', '标签噪声比例 r1', 'Test Accuracy', subplot_col='r2')


def plot_existing_extrapolation():
    """复用仓库已有结果 code/results/simulation_summary.csv 中的 BIAM-H 外推实验"""
    path = os.path.join(PROJECT_ROOT, 'results', 'simulation_summary.csv')
    if not os.path.exists(path):
        print('[skip] 未找到 simulation_summary.csv')
        return
    df = pd.read_csv(path)
    ext = df[df.experiment == 'biam_h_extrapolation']
    if not len(ext):
        return
    methods = ['BIAM(hinge)', 'BIAM-H(constant)']
    seeds = sorted(ext.seed.unique())
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for m, c in zip(methods, [COLORS['BIAM'], COLORS['MLP']]):
        vals = [ext[(ext.variant == m) & (ext.seed == s)].mse.values[0] for s in seeds]
        ax.bar([f'seed={s}' for s in seeds], vals, width=0.4, label=m, color=c)
    ax.set_yscale('log')
    ax.set_ylabel('Test MSE (log)')
    ax.set_title('外推场景：训练域[-1,0.8] → 测试域[1.0,1.5]（y=3·x0）\nhinge vs 分段常数基（仓库已有结果）', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_existing_extrapolation.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig] {out}')


if __name__ == '__main__':
    main()
    plot_existing_extrapolation()
