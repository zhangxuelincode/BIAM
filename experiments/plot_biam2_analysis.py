"""
BIAM² 交互分析图（论文 §6 新增可解释图）
============================================================================
图 1  fig_biam2_pair_mse    E6 各场景回归 MSE（BIAM2 vs BIAM vs 基线，log 轴）
图 2  fig_biam2_recovery    (a) 学习的成对交互强度热图（真值对框出）
                            (b) 各场景交互排序 AUPRC（虚线 = 随机水平）
图 3  fig_biam2_surface     真实交互曲面 vs BIAM² 拟合曲面（3 个植入对）

用法：
  python experiments/plot_biam2_analysis.py --seed 42
结果输出： exp/biam2/ 下 fig_*.{pdf,png} 与 diagnostics.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import argparse
import warnings
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp', 'biam2')
IMG_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'ICML26_Workshop_BIAM', 'images')

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False})

from run_biam2_experiments import (
    gen_interact_regression, TRUE_PAIRS, pair_auprc, run_biam2_full,
)
from run_new_experiments import DEVICE, set_seed

C_BIAM2 = '#c0392b'
C_BIAM = '#2c6fbb'


def save_fig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
    os.makedirs(IMG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(IMG_DIR, f'{name}.{ext}'),
                    bbox_inches='tight', dpi=300 if ext == 'png' else None)
    print(f'  [saved] {name}.pdf / .png', flush=True)


def save_diag(diag):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, 'diagnostics.json')
    old = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            old = json.load(f)
    old.update(diag)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(old, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# 图 1：E6 回归 MSE 分组条形图
# ----------------------------------------------------------------------------
def fig_pair_mse(seed=42):
    import pandas as pd
    csv = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp', 'results_biam2_E6.csv')
    df = pd.read_csv(csv)
    df = df[df.seed == seed] if (df.seed == seed).any() else df
    agg = df.groupby(['scenario', 'method'])['mse'].mean().reset_index()
    scen_order = ['int_clean', 'int_mcar', 'int_mar', 'int_mnar', 'null_main']
    labels = {'int_clean': 'Interactions,\nclean', 'int_mcar': 'Interactions,\nMCAR 20%',
              'int_mar': 'Interactions,\nMAR 20%', 'int_mnar': 'Interactions,\nMNAR 20%',
              'null_main': 'No interactions\n(MCAR 20%)'}
    methods = ['BIAM2', 'BIAM', 'Lasso', 'SAM', 'NAM', 'MLP', 'XGBoost', 'EBM', 'CatBoost']
    colors = [C_BIAM2, C_BIAM] + ['#9aa5b1'] * (len(methods) - 2)

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    W = 0.09
    for mi, meth in enumerate(methods):
        vals = []
        for sc in scen_order:
            sub = agg[(agg.scenario == sc) & (agg.method == meth)]
            vals.append(sub.mse.values[0] if len(sub) else np.nan)
        xs = np.arange(len(scen_order)) + (mi - (len(methods) - 1) / 2) * W
        ax.bar(xs, vals, width=W * 0.92, color=colors[mi],
               label=meth if meth in ('BIAM2', 'BIAM') else None,
               zorder=3 if meth in ('BIAM2', 'BIAM') else 2)
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(scen_order)))
    ax.set_xticklabels([labels[s] for s in scen_order], fontsize=8)
    ax.set_ylabel('test MSE (log)')
    ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    return fig


# ----------------------------------------------------------------------------
# 图 2：交互强度热图 + AUPRC
# ----------------------------------------------------------------------------
def fig_recovery(seed=42):
    from run_biam2_experiments import gen_interact_classification
    import pandas as pd
    csv = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp', 'results_biam2_E6.csv')
    df = pd.read_csv(csv)
    df = df[(df.method == 'BIAM2') & (df.seed == seed)]
    auprc_by_scen = dict(zip(df.scenario, df.auprc_pair))
    random_prev = 3 / 45

    # 训练 BIAM2（int_mcar, 指定种子）提取强度矩阵
    # E6/E7 使用发现档门控调度（与 §7 表格一致），见 0-PAPER.tex §7 协议
    set_seed(seed)
    (tr, va, te), pairs = gen_interact_regression('int_mcar', seed)
    _, model_metrics = None, None
    metrics, S = run_biam2_full('regression', tr, va, te, seed,
                                pair_l0=0.0, pair_l2=0.0, pair_warmup=0, gate_lr_scale=1.0)
    np.fill_diagonal(S, 0)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9),
                             gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axes[0]
    im = ax.imshow(S, cmap='viridis')
    for (j, k) in TRUE_PAIRS:
        ax.add_patch(Rectangle((k - 0.5, j - 0.5), 1, 1, fill=False,
                               edgecolor='w', lw=1.4))
        ax.add_patch(Rectangle((j - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor='w', lw=1.4))
    ax.set_xlabel('feature index $k$')
    ax.set_ylabel('feature index $j$')
    ax.set_title('(a) learned pair strengths $s_{jk}$', fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    vmax = float(S.max())
    cbar.set_ticks([0.0, vmax / 2, vmax])
    cbar.set_ticklabels(['0', f'{vmax / 2:.2f}', f'{vmax:.2f}'])

    ax = axes[1]
    scen_order = ['int_clean', 'int_mcar', 'int_mar', 'int_mnar', 'null_main']
    labels = ['clean', 'MCAR', 'MAR', 'MNAR', 'null']
    vals = [auprc_by_scen.get(s, np.nan) for s in scen_order]
    bars = ax.bar(labels, vals, color=[C_BIAM2] * 4 + ['#9aa5b1'], width=0.55)
    ax.axhline(random_prev, color='k', ls='--', lw=0.9)
    ax.set_xlim(-1.1, 4.55)
    ax.text(-1.02, random_prev + 0.018, 'random', fontsize=7.5, ha='left')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}',
                ha='center', fontsize=7.5)
    ax.set_ylabel('interaction-ranking AUPRC')
    ax.set_ylim(0, 1.05)
    ax.set_title('(b) recovery of planted pairs', fontsize=9)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    return fig, S, metrics


# ----------------------------------------------------------------------------
# 图 3：真实 vs 拟合交互曲面
# ----------------------------------------------------------------------------
def true_surface(name, gj, gk):
    gj = np.asarray(gj, dtype=np.float32)
    gk = np.asarray(gk, dtype=np.float32)
    GJ, GK = np.meshgrid(gj, gk, indexing='ij')
    if name == 'prod':
        return 2.0 * GJ * GK
    if name == 'sin':
        return 1.5 * np.sin(GJ * GK)
    if name == 'tanh':
        return 1.2 * np.tanh(GJ) * np.tanh(GK)
    raise ValueError(name)


def fig_surface(seed=42, grid_n=41):
    from run_biam2_experiments import gen_interact_regression
    from models.biam_model import BIAMModel
    from models.biam_weighting_network import BIAMWeightingNetwork
    from gradients.biam_optimizer import BIAMOptimizer
    from utils.biam_config import BIAMConfig
    import torch.utils.data as Data

    set_seed(seed)
    (tr, va, te), pairs = gen_interact_regression('int_mcar', seed)
    X_train, y_train = tr
    cfg = BIAMConfig(); cfg.task = 'regression'; cfg.epochs = 300; cfg.batch_size = 64
    cfg.upper_lr = 0.01; cfg.lower_lr = 0.05; cfg.n_knots = 10; cfg.seed = seed
    cfg.device = DEVICE; cfg.use_feature_interactions = True; cfg.n_pair_ranks = 8
    # 发现档门控调度（与 §7 表格一致）
    cfg.pair_warmup_epochs = 0; cfg.pair_gate_lr_scale = 1.0
    cfg.lambda_pair_l0 = 0.0; cfg.lambda_pair_l2 = 0.0
    cfg.input_dim = X_train.shape[1]
    tl = Data.DataLoader(Data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)), batch_size=64, shuffle=True)
    vl = Data.DataLoader(Data.TensorDataset(
        torch.tensor(va[0], dtype=torch.float32),
        torch.tensor(va[1], dtype=torch.float32)), batch_size=64, shuffle=False)
    m = BIAMModel(cfg, DEVICE); wn = BIAMWeightingNetwork(cfg, DEVICE)
    m.additive_model.set_knots(torch.tensor(X_train, dtype=torch.float32))
    op = BIAMOptimizer(cfg, m, wn)
    for ep in range(cfg.epochs):
        op.train_epoch(tl, vl, ep)

    pairs_meta = [(4, 5, 'prod', '$2.0\\,x_j x_k$'),
                  (6, 7, 'sin', '$1.5\\,\\sin(x_j x_k)$'),
                  (8, 9, 'tanh', '$1.2\\,\\tanh(x_j)\\tanh(x_k)$')]
    # 网格取 [-1.6, 1.6]（标准化后观测范围附近）
    g = np.linspace(-1.6, 1.6, grid_n)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    for ci, (j, k, name, latex) in enumerate(pairs_meta):
        Zt = true_surface(name, g, g)
        Zf = m.additive_model.get_pair_surface(j, k, g).astype(float)
        # 常数与加性边缘和主效应不可区分，从成对曲面中剥离（真值曲面均为零均值
        # 且边缘对称，剥离不改变真值）；随后按标准差对齐尺度
        Zf = Zf - Zf.mean()
        Zf = Zf - Zf.mean(axis=0, keepdims=True) - Zf.mean(axis=1, keepdims=True) + Zf.mean()
        scale = np.std(Zt) / max(np.std(Zf), 1e-8)
        Zf = Zf * scale
        vmin, vmax = -abs(Zt).max(), abs(Zt).max()
        ax = axes[0, ci]
        ax.imshow(Zt, origin='lower', extent=[g[0], g[-1], g[0], g[-1]],
                  cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'true  {latex}', fontsize=8.5)
        ax = axes[1, ci]
        ax.imshow(Zf, origin='lower', extent=[g[0], g[-1], g[0], g[-1]],
                  cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'BIAM$^2$ fitted ($x_{{{j+1}}}, x_{{{k+1}}}$)', fontsize=8.5)
        if ci == 0:
            axes[0, ci].set_ylabel('$x_j$')
            axes[1, ci].set_ylabel('$x_j$')
        for r in (0, 1):
            axes[r, ci].set_xlabel('$x_k$', fontsize=8)
    fig.tight_layout()

    mse = op.evaluate((te[0], te[1]))['mse']
    return fig, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f'[biam2] device: {DEVICE}', flush=True)
    diag = {'seed': args.seed}

    fig = fig_pair_mse(args.seed)
    save_fig(fig, 'fig_biam2_pair_mse')
    plt.close(fig)

    fig, S, metrics = fig_recovery(args.seed)
    save_fig(fig, 'fig_biam2_recovery')
    plt.close(fig)
    diag['recovery_int_mcar_mse'] = metrics['mse']
    flat = np.argsort(-S, axis=None)
    ij = np.dstack(np.unravel_index(flat, S.shape))[0]
    diag['top_pairs'] = [(int(i), int(k), round(float(S[i, k]), 4))
                         for i, k in ij if i < k][:6]

    fig, mse = fig_surface(args.seed)
    save_fig(fig, 'fig_biam2_surface')
    plt.close(fig)
    diag['surface_int_mcar_mse'] = mse
    save_diag(diag)
    print('[biam2] all figures done', flush=True)


if __name__ == '__main__':
    main()
