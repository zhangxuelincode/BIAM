# -*- coding: utf-8 -*-
"""
可解释性分析（Interpretability Analysis）
============================================================================
对论文 §6（合成压力测试）与 §5（真实基准）补充可解释性可视化：
  Fig.1  fig_interp_main_clean   干净数据下 8 个信息特征的主效应曲线：真值 vs BIAM vs NAM
  Fig.2  fig_interp_main_outlier 10% 标签离群下同样的形状恢复对比（鲁棒性不牺牲可解释性）
  Fig.3  fig_interp_missing      MNAR 20% 下缺失偏移的恢复（真值 vs 拟合）+ 缺失-特征
                                 条件响应曲面（真值 vs BIAM）
  Fig.4  fig_interp_real         Adult 真实数据：主要连续特征的贡献曲线 + 缺失指示效应

所有图同时输出 PDF（论文用）与 PNG（预览用）。

用法：
  python experiments/interpretability_analysis.py [--epochs 100] [--seed 42]
输出： exp/interpretability/ 下的 4 组 fig_*.{pdf,png} 与 diagnostics.json
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # Windows OpenMP 兼容
import sys
import json
import zlib
import argparse
import warnings
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_config import BIAMConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp', 'interpretability')

# 论文 §6.1 的真实加性成分（与 data/biam_data_generator.py 一致）
TRUE_TERMS = [
    lambda x: -2 * np.sin(2 * x),
    lambda x: 8 * np.square(x),
    lambda x: 7 * np.sin(x) / (2 - np.sin(x)),
    lambda x: 6 * np.exp(-x),
    lambda x: np.power(x, 3) + 1.5 * np.square(x - 1),
    lambda x: 5 * x,
    lambda x: 10 * np.sin(np.exp(-x / 2)),
    lambda x: -10 * norm.cdf(x, loc=0.5, scale=0.8),
]

C_TRUE, C_BIAM, C_NAM = '#2166AC', '#B2182B', '#8A8A8A'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                     'xtick.direction': 'out', 'ytick.direction': 'out'})


# ----------------------------------------------------------------------------
# 数据（与 data/biam_data_generator.py 的合成回归管道逐行一致，额外保留 scaler）
# ----------------------------------------------------------------------------
def make_synthetic_regression(mech='none', seed=42, n=1200, p=10, missing_ratio=0.2):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size=(n, p))
    y = np.zeros(n)
    for j in range(min(p, 8)):
        y = y + TRUE_TERMS[j](X[:, j])
    y = y + 0.1 * np.random.randn(n)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    sx = StandardScaler().fit(X_train)
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    Xtr, Xva, Xte = sx.transform(X_train), sx.transform(X_val), sx.transform(X_test)
    ytr = sy.transform(y_train.reshape(-1, 1)).flatten()
    yva = sy.transform(y_val.reshape(-1, 1)).flatten()
    yte = sy.transform(y_test.reshape(-1, 1)).flatten()
    if mech in ('MCAR', 'MAR', 'MNAR'):
        r = missing_ratio
        for Xa in (Xtr, Xva, Xte):
            if mech == 'MCAR':
                m = np.random.uniform(0, 1, Xa.shape) < r
                Xa[m] = np.nan
            elif mech == 'MAR':
                for j in range(p):
                    cond = (j + 1) % p
                    probs = 1.0 / (1.0 + np.exp(-(Xa[:, cond] - np.median(Xa[:, cond])) * 2))
                    mm = np.random.uniform(0, 1, Xa.shape[0]) < probs * r * 2
                    Xa[mm, j] = np.nan
            else:
                for j in range(p):
                    col = Xa[:, j]
                    probs = 1.0 / (1.0 + np.exp(-(col - np.median(col)) * 2))
                    mm = np.random.uniform(0, 1, Xa.shape[0]) < probs * r * 2
                    Xa[mm, j] = np.nan
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, sx=sx, sy=sy)


def apply_label_noise(y_train, scenario, seed):
    """确定性摘要替代 hash()，保证跨进程可复现（协议与 run_new_experiments.py 一致）"""
    tag = int(zlib.crc32(scenario.encode('utf-8')) % 9999)
    rng = np.random.default_rng(1000 * seed + tag)
    y = y_train.copy()
    n = len(y)
    if scenario in ('outlier10', 'outlier30'):
        r = 0.1 if scenario == 'outlier10' else 0.3
        idx = rng.choice(n, max(1, int(n * r)), replace=False)
        y[idx] += rng.normal(100, 100, len(idx))
    elif scenario in ('mixture_A', 'mixture_B'):
        shift = 8 if scenario == 'mixture_A' else 20
        comp = rng.random(n) < 0.8
        y += np.where(comp, rng.normal(-2 if scenario == 'mixture_A' else 0, 1, n),
                      rng.normal(shift, 1, n))
    else:
        raise ValueError(scenario)
    return y


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def train_biam(d, seed=42, epochs=100, task='regression', ytr_override=None):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = BIAMConfig(); cfg.task = task; cfg.seed = seed
    cfg.epochs = epochs; cfg.batch_size = 64
    cfg.upper_lr = 0.1 if task == 'classification' else 0.01
    cfg.lower_lr = 0.05; cfg.n_knots = 10
    cfg.device = DEVICE
    cfg.input_dim = d['Xtr'].shape[1]
    if task == 'classification':
        cfg.num_classes = 2

    import torch.utils.data as Data
    ytr = d['ytr'] if ytr_override is None else ytr_override
    tr_loader = Data.DataLoader(Data.TensorDataset(
        torch.tensor(d['Xtr'], dtype=torch.float32),
        torch.tensor(ytr, dtype=torch.float32)), batch_size=64, shuffle=True)
    va_loader = Data.DataLoader(Data.TensorDataset(
        torch.tensor(d['Xva'], dtype=torch.float32),
        torch.tensor(d['yva'], dtype=torch.float32)), batch_size=64, shuffle=False)
    model = BIAMModel(cfg, DEVICE)
    wn = BIAMWeightingNetwork(cfg, DEVICE)
    model.additive_model.set_knots(torch.tensor(d['Xtr'], dtype=torch.float32))
    opt = BIAMOptimizer(cfg, model, wn)
    for ep in range(epochs):
        opt.train_epoch(tr_loader, va_loader, ep)
    metrics = opt.evaluate((d['Xte'], d['yte']))
    return model, metrics


def main_effect_curves(am, grid_np, out_dim=0):
    """f_j(z) = Σ_τ B[j,τ,d]·relu(z−η_τ)，返回 (p, G) numpy"""
    with torch.no_grad():
        z = torch.tensor(grid_np, dtype=torch.float32, device=am.knots.device)
        Hb = torch.relu(z.view(-1, 1) - am.knots.view(1, -1))            # (G, L)
        cur = torch.einsum('gl,pld->pgd', Hb, am.B)[..., out_dim]
        return cur.cpu().numpy()


def gate_response_curve(am, j, k, grid_np, out_dim=0):
    """缺失-特征交互响应 Δ_{j,k}(z) = Σ_τ α_{j,k,τ}·relu(z−η_τ)"""
    with torch.no_grad():
        z = torch.tensor(grid_np, dtype=torch.float32, device=am.knots.device)
        Hb = torch.relu(z.view(-1, 1) - am.knots.view(1, -1))            # (G, L)
        cur = Hb @ am.A[j, k, :, out_dim]
        return cur.cpu().numpy()


# ----------------------------------------------------------------------------
# NAM 基线（与 run_new_experiments.py 相同结构）
# ----------------------------------------------------------------------------
class NAMNet(torch.nn.Module):
    def __init__(self, p, hidden=32):
        super().__init__()
        self.nets = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(1, hidden), torch.nn.ReLU(),
                                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                torch.nn.Linear(hidden, 1)) for _ in range(p)])
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.bias.view(1, -1)
        for j, net in enumerate(self.nets):
            out = out + net(x[:, j:j + 1])
        return out


def train_nam(d, seed=42, epochs=300, ytr_override=None):
    np.random.seed(seed); torch.manual_seed(seed)
    Xtr = torch.tensor(d['Xtr'], dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(d['ytr'] if ytr_override is None else ytr_override,
                       dtype=torch.float32, device=DEVICE)
    net = NAMNet(d['Xtr'].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xtr[idx]).squeeze(-1), ytr[idx])
            loss.backward()
            opt.step()
    net.eval()
    return net


@torch.no_grad()
def nam_main_curves(net, grid_np):
    z = torch.tensor(grid_np, dtype=torch.float32, device=DEVICE).view(-1, 1)
    return np.stack([net.nets[j](z).squeeze(-1).cpu().numpy()
                     for j in range(len(net.nets))])


# ----------------------------------------------------------------------------
# 真值曲线（标准化 y 单位，中心化）
# ----------------------------------------------------------------------------
def truth_curve(j, d, grid_np, center=True):
    mu_j, s_j = d['sx'].mean_[j], d['sx'].scale_[j]
    g = TRUE_TERMS[j](mu_j + s_j * grid_np) / d['sy'].scale_[0]
    return g - g.mean() if center else g


def save_fig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 论文 images/ 目录：所有图同步输出 PDF+PNG，供 LaTeX 正文引用
    img_dir = os.path.join(os.path.dirname(PROJECT_ROOT), 'ICML26_Workshop_BIAM', 'images')
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, bbox_inches='tight', dpi=300 if ext == 'png' else None)
        try:
            os.makedirs(img_dir, exist_ok=True)
            fig.savefig(os.path.join(img_dir, f'{name}.{ext}'),
                        bbox_inches='tight', dpi=300 if ext == 'png' else None)
        except OSError:
            pass
    plt.close(fig)
    print(f'  [saved] {name}.pdf / .png', flush=True)


# ----------------------------------------------------------------------------
# Fig.1 / Fig.2：主效应形状恢复
# ----------------------------------------------------------------------------
def fig_main_effects(d, model_biam, net_nam, name, mse_biam, title_tag):
    grid = np.linspace(-2.2, 2.2, 400)
    fit = main_effect_curves(model_biam.additive_model, grid)
    nam = nam_main_curves(net_nam, grid)
    sy = d['sy'].scale_[0]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.4), sharex=True)
    corrs = []
    for j, ax in enumerate(axes.flat):
        t = truth_curve(j, d, grid)
        # 模型在标准化 y 上训练，其输出已是标准化单位；真值需除以 sy 对齐
        fb = fit[j] - fit[j].mean()
        fn = nam[j] - nam[j].mean()
        # 以真值的标准差为尺度基准，保证三者的纵向比例可比
        scale = np.std(t)
        t_n = t / scale; fb_n = fb / scale; fn_n = fn / scale
        corrs.append(np.corrcoef(t, fb)[0, 1])
        ax.plot(grid, t_n, color=C_TRUE, lw=1.4, label='True')
        ax.plot(grid, fb_n, color=C_BIAM, lw=1.2, ls='--', label=f'BIAM (MSE {mse_biam:.3f})')
        ax.plot(grid, fn_n, color=C_NAM, lw=1.0, ls=':', label='NAM')
        ax.set_title(f'$x_{{{j+1}}}$', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylim(-2.1, 2.1)
    for ax in axes[1]:
        ax.set_xlabel('feature value (standardized)', fontsize=8)
    axes[0, 0].set_ylabel('contribution', fontsize=8)
    axes[1, 0].set_ylabel('contribution', fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:1] + [handles[1], handles[2]],
               [labels[0], f'BIAM (MSE {mse_biam:.3f})', labels[2]],
               loc='upper center', ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save_fig(fig, name)
    return corrs


# ----------------------------------------------------------------------------
# Fig.3：MNAR 缺失偏移恢复 + 缺失-特征条件响应曲面
# ----------------------------------------------------------------------------
def mnar_true_offsets(d, seed=42, missing_ratio=0.2):
    """φ*_j = E[g_j|m_j=1] − E[g_j|m_j=0]（标准化 y 单位）。
    权重 w(z) = sigmoid(2z)（比例 r 在两个条件期望中抵消）。"""
    z = np.linspace(-3.5, 3.5, 4001)
    w = 1.0 / (1.0 + np.exp(-2 * z))
    out = np.zeros(8)
    for j in range(8):
        mu_j, s_j = d['sx'].mean_[j], d['sx'].scale_[j]
        zlo, zhi = (-1.0 - mu_j) / s_j, (1.0 - mu_j) / s_j
        m = (z >= zlo) & (z <= zhi)
        zz, ww = z[m], w[m]
        g = TRUE_TERMS[j](mu_j + s_j * zz)
        E1 = np.trapezoid(g * ww, zz) / np.trapezoid(ww, zz)
        E0 = np.trapezoid(g * (1 - ww), zz) / np.trapezoid(1 - ww, zz)
        out[j] = (E1 - E0) / d['sy'].scale_[0]
    return out


def fitted_offsets(model, d):
    """模型级干预偏移：φ̂_j = mean_rows[f(x, x_j 置缺失) − f(x, x_j 观测)]。
    仅用全观测行做差分（其余特征的缺失偏移逐行抵消），自动包含 β^miss
    与缺失交互的全部贡献。"""
    am = model.additive_model
    Xnp = d['Xte']
    Xnp = Xnp[~np.isnan(Xnp).any(axis=1)]           # 全观测行
    X = torch.tensor(Xnp, dtype=torch.float32, device=am.knots.device)
    with torch.no_grad():
        base = am(X)                                # (n, d)
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            Xj = X.clone()
            Xj[:, j] = float('nan')                 # 将特征 j 置为缺失
            out[j] = (am(Xj) - base).mean().item()
    return out


def fig_missing(d, model, name):
    am = model.additive_model
    grid = np.linspace(-2.2, 2.2, 120)
    phi_star = mnar_true_offsets(d)
    phi_hat = fitted_offsets(model, d)

    # 曲面：D(z_j, z_k) = [x_j 缺失时的预测] − [x_j 观测为 z_j 时的预测]
    j_top = int(np.argmax(np.abs(phi_star[:8])))
    blk_norm = np.sqrt((am.A[j_top, :, :, 0] ** 2).sum(-1).cpu().detach().numpy())
    k_top = int(np.argmax(blk_norm))
    f_top = main_effect_curves(am, grid)[j_top]
    dg = gate_response_curve(am, j_top, k_top, grid)
    dj = am.beta_miss[j_top, 0].item()
    Zj, Zk = np.meshgrid(grid, grid, indexing='ij')
    D_hat = dj + dg[None, :] - f_top[:, None]
    D_true = phi_star[j_top] - truth_curve(j_top, d, grid, center=False)[:, None]
    # 中心化以消去截距的不可辨识常数
    D_hat -= D_hat.mean(); D_true -= D_true.mean()
    vmax = max(np.abs(D_hat).max(), np.abs(D_true).max())

    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(8)
    ax.bar(xs - 0.18, phi_star, width=0.36, color=C_TRUE, label=r'true $\phi^{*}_j$')
    ax.bar(xs + 0.18, phi_hat[:8], width=0.36, color=C_BIAM, label=r'BIAM $\hat{\phi}_j$')
    ax.set_xticks(xs); ax.set_xticklabels([f'{i+1}' for i in xs], fontsize=7)
    ax.set_xlabel('feature index $j$', fontsize=8)
    ax.set_ylabel(r'missing offset (std. $y$)', fontsize=8)
    ax.axhline(0, color='k', lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    ax.set_title('(a) offsets under MNAR', fontsize=9)

    for col_i, (Z, ttl) in enumerate([(D_true, '(b) Bayes surface'),
                                      (D_hat, '(c) BIAM surface')]):
        ax3 = fig.add_subplot(gs[0, col_i + 1], projection='3d')
        ax3.plot_surface(Zj, Zk, Z, cmap='viridis', rstride=2, cstride=2,
                         vmin=-vmax, vmax=vmax, linewidth=0, antialiased=True, alpha=0.95)
        ax3.set_xlabel('$z_j$', fontsize=7, labelpad=-4)
        ax3.set_ylabel('$z_k$', fontsize=7, labelpad=-4)
        ax3.tick_params(labelsize=6, pad=-2)
        ax3.set_zlim(-vmax, vmax)
        ax3.set_title(f'{ttl}', fontsize=9)
        ax3.view_init(elev=22, azim=-60)

    fig.tight_layout()
    save_fig(fig, name)
    return dict(j_top=j_top, k_top=k_top, phi_star=phi_star.tolist(),
                phi_hat=phi_hat[:8].tolist())


# ----------------------------------------------------------------------------
# Fig.4：Adult 真实数据
# ----------------------------------------------------------------------------
CONTINUOUS = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


def load_adult(seed=42, n_train=6000, n_test=2000):
    import pandas as pd
    from sklearn.datasets import fetch_openml
    adult = fetch_openml('adult', version=2, as_frame=True)
    X, y = adult.data, adult.target
    y = (y.astype(str) == '>50K').astype(np.float32)
    feature_names = list(X.columns)
    X = X.copy()
    nan_mask = np.zeros(X.shape, dtype=bool)
    for ci, c in enumerate(X.columns):
        col = X[c]
        if col.dtype == object or str(col.dtype) == 'category':
            s = col.astype(str)
            isna = s == '?'                     # adult 用 '?' 表示缺失
            nan_mask[:, ci] |= isna.values
            le = LabelEncoder()
            le.fit(s[~isna])
            codes = pd.Series(np.nan, index=X.index)
            codes[~isna] = le.transform(s[~isna])
            X[c] = codes
        else:
            nan_mask[:, ci] |= pd.isna(col).values
    X = np.asarray(X.values, dtype=np.float32)
    # 统一 winsorize 重尾列，避免 capital-gain 的长尾支配标准化
    for c in range(X.shape[1]):
        col = X[:, c]
        v = col[~np.isnan(col)]
        if len(v) and np.quantile(v, 0.99) > 10 * (np.median(v) + 1):
            cap = np.quantile(v, 0.99)
            col[col > cap] = cap
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=seed, stratify=ytmp)
    # 转 numpy，避免 pandas 按标签索引导致 rng.choice 的位置索引失败
    ytr, ytmp, yva, yte = (np.asarray(a) for a in (ytr, ytmp, yva, yte))
    sx = StandardScaler().fit(Xtr)
    Xtr_s = sx.transform(Xtr); Xva_s = sx.transform(Xva); Xte_s = sx.transform(Xte)
    # 缺失位置在标准化后还原为 NaN（nan_mask 同步切分）
    nm_tr, nm_tmp = nan_mask[:len(Xtr)], nan_mask[len(Xtr):]
    nm_va, nm_te = nm_tmp[:len(Xva)], nm_tmp[len(Xva):]
    Xtr_s[nm_tr] = np.nan; Xva_s[nm_va] = np.nan; Xte_s[nm_te] = np.nan
    # 子采样
    rng = np.random.default_rng(seed)
    it = rng.choice(len(Xtr_s), min(n_train, len(Xtr_s)), replace=False)
    ie = rng.choice(len(Xte_s), min(n_test, len(Xte_s)), replace=False)
    return dict(Xtr=Xtr_s[it], ytr=ytr[it], Xva=Xva_s, yva=yva,
                Xte=Xte_s[ie], yte=yte[ie], sx=sx, names=feature_names)


@torch.no_grad()
def clf_metrics(model, d):
    Xt = torch.tensor(d['Xte'], dtype=torch.float32, device=DEVICE)
    logits = model(Xt).cpu().numpy()
    pred = (logits[:, 1] > logits[:, 0]).astype(int)
    return dict(accuracy=float(accuracy_score(d['yte'], pred)),
                f1_macro=float(f1_score(d['yte'], pred, average='macro', zero_division=0)))


def fig_real(d, model, name, metrics):
    am = model.additive_model
    grid = np.linspace(-2.2, 2.2, 400)
    curves = main_effect_curves(am, grid, out_dim=1)  # income>50K 的 logit 贡献
    imp = np.sqrt((am.B ** 2).sum(dim=(1, 2)).detach().cpu().numpy())
    order = np.argsort(-imp)

    cont_idx = [d['names'].index(c) for c in CONTINUOUS if c in d['names']]
    pick = [j for j in order if j in cont_idx][:6]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.4), sharex=True)
    for ax, j in zip(axes.flat, pick):
        c = curves[j] - curves[j].mean()
        ax.plot(grid, c, color=C_BIAM, lw=1.2)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(d['names'][j], fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[1]:
        ax.set_xlabel('feature value (standardized)', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\hat{f}_j$ (logit)', fontsize=8)
    fig.suptitle(f"Adult: feature contributions to the income logit "
                 f"(test acc {metrics['accuracy']:.3f}, macro-F1 {metrics['f1_macro']:.3f})",
                 fontsize=9)
    fig.tight_layout()
    save_fig(fig, name)
    return dict(picked=[d['names'][j] for j in pick])


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def save_diag(diag):
    """增量写入诊断（支持分步生成各图而不互相覆盖）"""
    diag_path = os.path.join(OUT_DIR, 'diagnostics.json')
    old = {}
    if os.path.exists(diag_path):
        with open(diag_path) as f:
            try:
                old = json.load(f)
            except Exception:
                old = {}
    with open(diag_path, 'w') as f:
        json.dump({**old, **diag}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-real', action='store_true')
    parser.add_argument('--skip-synth', action='store_true')
    args = parser.parse_args()
    seed = args.seed
    diag = {}

    print(f'Device: {DEVICE}', flush=True)

    if not args.skip_synth:
        # --- Fig.1 clean ---
        print('=== Fig.1 clean ===', flush=True)
        d = make_synthetic_regression('none', seed=seed)
        model_c, m_c = train_biam(d, seed=seed, epochs=args.epochs)
        nam_c = train_nam(d, seed=seed)
        corrs = fig_main_effects(d, model_c, nam_c, 'fig_interp_main_clean', m_c['mse'], 'Clean data')
        diag['clean'] = dict(mse=float(m_c['mse']), corr=corrs)
        print(f"  clean mse={m_c['mse']:.4f}, corr={np.round(corrs,3)}", flush=True)
        save_diag(diag)

        # --- Fig.2 outlier10 ---
        print('=== Fig.2 outlier10 ===', flush=True)
        d2 = make_synthetic_regression('none', seed=seed)
        y_out = apply_label_noise(d2['ytr'], 'outlier10', seed)
        model_o, m_o = train_biam(d2, seed=seed, epochs=args.epochs, ytr_override=y_out)
        nam_o = train_nam(d2, seed=seed, ytr_override=y_out)
        corrs_o = fig_main_effects(d2, model_o, nam_o, 'fig_interp_main_outlier', m_o['mse'],
                                   '10% label outliers')
        diag['outlier10'] = dict(mse=float(m_o['mse']), corr=corrs_o)
        print(f"  outlier10 mse={m_o['mse']:.4f}, corr={np.round(corrs_o,3)}", flush=True)
        save_diag(diag)

        # --- Fig.3 MNAR20 ---
        print('=== Fig.3 MNAR20 ===', flush=True)
        d3 = make_synthetic_regression('MNAR', seed=seed)
        model_m, m_m = train_biam(d3, seed=seed, epochs=args.epochs)
        info = fig_missing(d3, model_m, 'fig_interp_missing')
        diag['mnar'] = dict(mse=float(m_m['mse']), **info)
        print(f"  MNAR20 mse={m_m['mse']:.4f}, j_top={info['j_top']}, k_top={info['k_top']}", flush=True)
        save_diag(diag)

    # --- Fig.4 Adult ---
    if not args.skip_real:
        print('=== Fig.4 Adult ===', flush=True)
        d4 = load_adult(seed=seed)
        model_r, m_r = train_biam(d4, seed=seed, epochs=80, task='classification')
        metrics = clf_metrics(model_r, d4)
        info_r = fig_real(d4, model_r, 'fig_interp_real', metrics)
        diag['adult'] = dict(metrics=metrics, **info_r)
        print(f"  Adult metrics={metrics}, picked={info_r['picked']}", flush=True)
        save_diag(diag)

    print('DONE', flush=True)


if __name__ == '__main__':
    main()
