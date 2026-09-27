"""
新增对比实验（参考 MAM / MLNAM / S2MAM 三篇论文的实验协议，不含半监督部分）
============================================================================
E1 合成回归噪声谱系（参考 MAM εA/εB/εC 与 MLNAM 离群噪声协议）：
   - clean / mixture_A(εA) / mixture_B(εB) / heavy_t(εC) / outlier10 / outlier30
E2 缺失机制对比（BIAM 缺失建模模块 vs 均值填充基线）：
   - MCAR/MAR/MNAR × ratio=0.2，以及 MCAR ratio∈{0.1,0.3}
E3 合成分类标签噪声×类失衡联合（参考 MAM 联合实验协议）：
   - r1∈{0.1,0.3} × r2∈{0.10,0.15}，测试集保持干净自然分布

对比方法：BIAM、BIAM-B（消融）、BIAM-I（消融）、Lasso、SAM(spline 加性)、
          NAM(神经加性)、MLP、XGBoost（基线缺失值统一均值填充，S2MAM 协议）
评估指标：回归 MSE/MAE；分类 ACC/Macro-F1；结果为 5 个随机种子 mean±std

用法：
  python experiments/run_new_experiments.py --blocks E1 E2 E3
  python experiments/run_new_experiments.py --smoke          # 快速计时冒烟测试
结果输出： exp/results_new_experiments.csv
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # Windows OpenMP 兼容
import sys
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.biam_data_generator import BIAMDataGenerator
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from utils.biam_config import BIAMConfig

from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import SplineTransformer
from scipy.sparse import hstack

SEEDS = [42, 7, 123, 2024, 3407]
# 超参数采用仓库 tests/test_biam_simulation.py 经校准的配置：
# batch=64, lower_lr=0.05, n_knots=10；分类 upper_lr=0.1；回归 upper_lr=0.01
EPOCHS_REG, EPOCHS_CLF = 100, 80
EXP_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp')


def result_csv(block: str) -> str:
    return os.path.join(EXP_DIR, f'results_new_{block}.csv')


def _existing_rows(block: str):
    """已完成的 (scenario, method, seed) 集合，用于断点续跑"""
    path = result_csv(block)
    done = set()
    if os.path.exists(path):
        df = pd.read_csv(path)
        done = set(zip(df.scenario, df.method, df.seed))
    return done


# --biam2-only：仅重跑 BIAM2 行（跳过基线），用于正则修复后的增量重跑
BIAM2_ONLY = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def impute_mean(X_train, *others):
    """均值填充（训练集 nanmean），S2MAM 协议中真实数据缺失处理方式"""
    mu = np.nanmean(X_train, axis=0)
    out = []
    for X in (X_train, *others):
        Xi = X.copy()
        idx = np.where(np.isnan(Xi))
        Xi[idx] = np.take(mu, idx[1])
        out.append(Xi)
    return out[0], *out[1:]


def append_row(row: dict):
    df = pd.DataFrame([row])
    path = result_csv(row['block'])
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)
    print(f"    [saved] {row['block']}/{row['scenario']}/{row['method']} seed={row['seed']} "
          f"metrics={ {k: round(v, 4) for k, v in row.items() if k in ('mse', 'mae', 'accuracy', 'f1_macro')} }",
          flush=True)


# ----------------------------------------------------------------------------
# BIAM 及其变体
# ----------------------------------------------------------------------------
def run_biam(task, train_data, val_data, test_data, seed, epochs=None,
             variant='BIAM', batch_size=64, pair_l0=None, pair_l2=None,
             pair_warmup=None, gate_lr_scale=None):
    """variant: BIAM | BIAM2 | BIAM-B | BIAM-I | BIAM-H
    超参数与 tests/test_biam_simulation.py 的校准配置一致"""
    if epochs is None:
        if variant == 'BIAM2':
            epochs = 300 if task == 'regression' else 200
        else:
            epochs = EPOCHS_REG if task == 'regression' else EPOCHS_CLF
    set_seed(seed)
    config = BIAMConfig()
    config.task = task
    config.epochs = epochs
    config.batch_size = batch_size
    config.upper_lr = 0.1 if task == 'classification' else 0.01
    config.lower_lr = 0.05
    config.n_knots = 10
    config.seed = seed
    config.device = DEVICE
    if variant == 'BIAM2':
        # BIAM²：二阶特征交互（校准配置：秩 8；回归 300 / 分类 200 epochs）
        config.use_feature_interactions = True
        config.n_pair_ranks = 8
        # BIAM² 成对门控的独立正则系数（None 时用 config 默认值）
        if pair_l0 is not None:
            config.lambda_pair_l0 = pair_l0
        if pair_l2 is not None:
            config.lambda_pair_l2 = pair_l2
        if pair_warmup is not None:
            config.pair_warmup_epochs = pair_warmup
        if gate_lr_scale is not None:
            config.pair_gate_lr_scale = gate_lr_scale
    if variant == 'BIAM-B':
        config.use_bilevel = False
    if variant == 'BIAM-I':
        config.use_missing_interactions = False
    if variant == 'BIAM-H':
        config.basis_type = 'piecewise_constant'

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data
    config.input_dim = X_train.shape[1]
    if task == 'classification':
        config.num_classes = 2

    import torch.utils.data as Data
    train_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                           torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False)

    biam_model = BIAMModel(config, DEVICE)
    weighting_network = BIAMWeightingNetwork(config, DEVICE)
    biam_model.additive_model.set_knots(torch.tensor(X_train, dtype=torch.float32))
    optimizer = BIAMOptimizer(config, biam_model, weighting_network)

    for epoch in range(epochs):
        optimizer.train_epoch(train_loader, val_loader, epoch)

    metrics = optimizer.evaluate(test_data)
    return metrics


# ----------------------------------------------------------------------------
# Torch 基线：NAM / MLP（GPU）
# ----------------------------------------------------------------------------
class NAMNet(nn.Module):
    """Neural Additive Model：每个特征一个子网络，输出求和（加性结构）"""
    def __init__(self, p, hidden=32, task='regression'):
        super().__init__()
        self.task = task
        self.nets = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1)) for _ in range(p)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.bias.view(1, -1)
        for j, net in enumerate(self.nets):
            out = out + net(x[:, j:j + 1])
        return out


class MLPNet(nn.Module):
    def __init__(self, p, task='regression'):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


def train_torch_baseline(model, X_train, y_train, task, seed, epochs=300, lr=1e-3, batch=256):
    set_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss() if task == 'regression' else nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xtr[idx]).squeeze(-1)
            loss = lossf(pred, ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_torch(model, X, task):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    pred = model(Xt).squeeze(-1).cpu().numpy()
    if task == 'classification':
        return (pred > 0).astype(int)
    return pred


# ----------------------------------------------------------------------------
# 传统基线：Lasso / SAM / XGBoost（均值填充输入）
# ----------------------------------------------------------------------------
def run_lasso(task, X_train, y_train, X_test, seed):
    if task == 'regression':
        model = LassoCV(cv=3, n_alphas=50, max_iter=5000, random_state=seed)
    else:
        model = LogisticRegressionCV(cv=3, penalty='l1', solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    if task == 'classification':
        return model.predict(X_test)
    return model.predict(X_test)


def run_sam(task, X_train, y_train, X_test, seed):
    """Sparse-Additive-Model 风格基线：逐特征 spline 变换 + (岭)线性模型"""
    set_seed(seed)
    p = X_train.shape[1]
    parts_tr, parts_te = [], []
    for j in range(p):
        st = SplineTransformer(n_knots=8, degree=3, include_bias=False)
        parts_tr.append(st.fit_transform(X_train[:, [j]]))
        parts_te.append(st.transform(X_test[:, [j]]))
    Xtr, Xte = np.hstack(parts_tr), np.hstack(parts_te)
    if task == 'regression':
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    else:
        model = LogisticRegressionCV(cv=3, penalty='l1', solver='liblinear', max_iter=1000)
    model.fit(Xtr, y_train)
    if task == 'classification':
        return model.predict(Xte)
    return model.predict(Xte)


def run_xgb(task, X_train, y_train, X_test, seed):
    from xgboost import XGBRegressor, XGBClassifier
    if task == 'regression':
        model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1,
                             tree_method='hist', n_jobs=4, random_state=seed)
    else:
        model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                              tree_method='hist', n_jobs=4, random_state=seed,
                              eval_metric='logloss')
    model.fit(X_train, np.asarray(y_train).astype(int) if task == 'classification' else y_train)
    return model.predict(X_test)


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def reg_metrics(y_true, y_pred):
    e = np.asarray(y_pred) - np.asarray(y_true)
    mse = float(np.mean(e ** 2))
    return {'mse': mse, 'mae': float(np.mean(np.abs(e)))}


def clf_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)}


# ----------------------------------------------------------------------------
# E1: 合成回归 · 噪声谱系（MAM εA/εB/εC + MLNAM 离群协议）
# ----------------------------------------------------------------------------
E1_SCENARIOS = ['clean', 'mixture_A', 'mixture_B', 'heavy_t', 'outlier10', 'outlier30']


def apply_regression_noise(scenario, y_train, seed):
    """噪声仅作用于训练集标签（标准化后的 y）。εA/εB/εC 作用于全部训练样本（MAM 协议）；
    outlier 按比例 r 对训练样本叠加 N(100,100)（MLNAM 协议）
    注意：用 zlib.crc32 而非 hash()——Windows 下 Python 字符串 hash 每进程随机化，
    会导致同一 seed 在不同进程/运行中产生不同的噪声实现，破坏跨运行可复现性。"""
    import zlib
    rng = np.random.default_rng(1000 * seed + zlib.crc32(scenario.encode('utf-8')) % 9999)
    y = y_train.copy()
    n = len(y)
    if scenario == 'clean':
        pass
    elif scenario == 'mixture_A':      # εA：偏态零均值混合
        comp = rng.random(n) < 0.8
        y += np.where(comp, rng.normal(-2, 1, n), rng.normal(8, 1, n))
    elif scenario == 'mixture_B':      # εB：偏态混合，离群簇在 20
        comp = rng.random(n) < 0.8
        y += np.where(comp, rng.normal(0, 1, n), rng.normal(20, 1, n))
    elif scenario == 'heavy_t':        # εC：自由度 2 的重尾 t 分布
        y += rng.standard_t(df=2, size=n)
    elif scenario in ('outlier10', 'outlier30'):
        r = 0.1 if scenario == 'outlier10' else 0.3
        idx = rng.choice(n, max(1, int(n * r)), replace=False)
        y[idx] += rng.normal(100, 100, len(idx))
    else:
        raise ValueError(scenario)
    return y


def block_E1(seeds):
    task = 'regression'
    done = _existing_rows('E1')
    for scenario in E1_SCENARIOS:
        for seed in seeds:
            # 数据：n=1200, p=10（Feng 8 信息特征 + 2 无关特征），无缺失
            set_seed(seed)
            cfg = BIAMConfig(); cfg.task = task; cfg.seed = seed
            cfg.noise_ratio = 0; cfg.missing_ratio = 0
            gen = BIAMDataGenerator(cfg)
            _, _, train_data, val_data, test_data = gen.generate_data()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data

            y_train_noisy = apply_regression_noise(scenario, y_train, seed)
            train_noisy = (X_train, y_train_noisy)

            # --- BIAM 家族（GPU）---
            for variant in (('BIAM2',) if BIAM2_ONLY else ('BIAM2', 'BIAM', 'BIAM-B')):
                if (scenario, variant, seed) in done:
                    continue
                t0 = time.time()
                m = run_biam(task, train_noisy, val_data, test_data, seed, variant=variant)
                append_row({'block': 'E1', 'scenario': scenario, 'method': variant, 'seed': seed,
                            'time_sec': round(time.time() - t0, 1), **m})

            # --- 基线（均值填充）---
            if not BIAM2_ONLY:
                Xtr_i, Xte_i = impute_mean(X_train, X_test)
                baselines = {
                    'Lasso': run_lasso(task, Xtr_i, y_train_noisy, Xte_i, seed),
                    'SAM': run_sam(task, Xtr_i, y_train_noisy, Xte_i, seed),
                    'XGBoost': run_xgb(task, Xtr_i, y_train_noisy, Xte_i, seed),
                }
                for name, net in (('NAM', NAMNet(X_train.shape[1], task=task)),
                                  ('MLP', MLPNet(X_train.shape[1], task=task))):
                    t0 = time.time()
                    net = net.to(DEVICE)
                    net = train_torch_baseline(net, Xtr_i, y_train_noisy, task, seed)
                    baselines[name] = predict_torch(net, Xte_i, task)
                    append_row({'block': 'E1', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': round(time.time() - t0, 1),
                                **reg_metrics(y_test, baselines[name])})
                for name, pred in baselines.items():
                    if name in ('NAM', 'MLP'):
                        continue
                    append_row({'block': 'E1', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': np.nan, **reg_metrics(y_test, pred)})


# ----------------------------------------------------------------------------
# E2: 缺失机制 × 缺失比例（BIAM 缺失建模 vs 均值填充基线）
# ----------------------------------------------------------------------------
E2_SCENARIOS = [('MCAR', 0.1), ('MCAR', 0.2), ('MCAR', 0.3), ('MAR', 0.2), ('MNAR', 0.2)]


def block_E2(seeds):
    task = 'regression'
    done = _existing_rows('E2')
    for mech, ratio in E2_SCENARIOS:
        scenario = f'{mech}_{int(ratio*100)}'
        for seed in seeds:
            set_seed(seed)
            cfg = BIAMConfig(); cfg.task = task; cfg.seed = seed
            cfg.noise_ratio = 0.3; cfg.noise_mean, cfg.noise_std = 1.0, 0.5  # 30% 标签噪声，BIAM 主场景
            cfg.missing_ratio = ratio; cfg.missing_mechanism = mech
            gen = BIAMDataGenerator(cfg)
            _, _, train_data, val_data, test_data = gen.generate_data()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data

            for variant in (('BIAM2',) if BIAM2_ONLY else ('BIAM2', 'BIAM', 'BIAM-I')):   # BIAM-I：移除缺失交互模块的消融
                if (scenario, variant, seed) in done:
                    continue
                t0 = time.time()
                m = run_biam(task, train_data, val_data, test_data, seed, variant=variant)
                append_row({'block': 'E2', 'scenario': scenario, 'method': variant, 'seed': seed,
                            'time_sec': round(time.time() - t0, 1), **m})

            if not BIAM2_ONLY:
                Xtr_i, Xte_i = impute_mean(X_train, X_test)
                preds = {
                    'Lasso': run_lasso(task, Xtr_i, y_train, Xte_i, seed),
                    'SAM': run_sam(task, Xtr_i, y_train, Xte_i, seed),
                    'XGBoost': run_xgb(task, Xtr_i, y_train, Xte_i, seed),
                }
                for name, net in (('NAM', NAMNet(X_train.shape[1], task=task)),
                                  ('MLP', MLPNet(X_train.shape[1], task=task))):
                    net = net.to(DEVICE)
                    net = train_torch_baseline(net, Xtr_i, y_train, task, seed)
                    preds[name] = predict_torch(net, Xte_i, task)
                    append_row({'block': 'E2', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': np.nan, **reg_metrics(y_test, preds[name])})
                for name, pred in preds.items():
                    if name in ('NAM', 'MLP'):
                        continue
                    append_row({'block': 'E2', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': np.nan, **reg_metrics(y_test, pred)})


# ----------------------------------------------------------------------------
# E3: 合成分类 · 标签噪声 × 类失衡（MAM 联合协议）
# ----------------------------------------------------------------------------
E3_SCENARIOS = [('clean', 0.0, 0.0), ('r1=0.1,r2=0.10', 0.1, 0.10),
                ('r1=0.3,r2=0.10', 0.3, 0.10), ('r1=0.3,r2=0.15', 0.3, 0.15),
                ('r1=0.5,r2=0.15', 0.5, 0.15)]


def block_E3(seeds):
    task = 'classification'
    done = _existing_rows('E3')
    for scenario, r1, r2 in E3_SCENARIOS:
        for seed in seeds:
            set_seed(seed)
            cfg = BIAMConfig(); cfg.task = task; cfg.seed = seed
            cfg.noise_ratio = r1; cfg.imbalance_ratio = r2; cfg.missing_ratio = 0
            gen = BIAMDataGenerator(cfg)
            _, _, train_data, val_data, test_data = gen.generate_data()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data

            for variant in (('BIAM2',) if BIAM2_ONLY else ('BIAM2', 'BIAM', 'BIAM-B')):
                if (scenario, variant, seed) in done:
                    continue
                t0 = time.time()
                m = run_biam(task, train_data, val_data, test_data, seed, variant=variant)
                append_row({'block': 'E3', 'scenario': scenario, 'method': variant, 'seed': seed,
                            'time_sec': round(time.time() - t0, 1), **m})

            if not BIAM2_ONLY:
                Xtr_i, Xte_i = impute_mean(X_train, X_test)
                preds = {
                    'Lasso': run_lasso(task, Xtr_i, y_train, Xte_i, seed),
                    'SAM': run_sam(task, Xtr_i, y_train, Xte_i, seed),
                    'XGBoost': run_xgb(task, Xtr_i, y_train, Xte_i, seed),
                }
                for name, net in (('NAM', NAMNet(X_train.shape[1], task=task)),
                                  ('MLP', MLPNet(X_train.shape[1], task=task))):
                    net = net.to(DEVICE)
                    net = train_torch_baseline(net, Xtr_i, y_train, task, seed)
                    preds[name] = predict_torch(net, Xte_i, task)
                    append_row({'block': 'E3', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': np.nan, **clf_metrics(y_test, preds[name])})
                for name, pred in preds.items():
                    if name in ('NAM', 'MLP'):
                        continue
                    append_row({'block': 'E3', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': np.nan, **clf_metrics(y_test, pred)})


# ----------------------------------------------------------------------------
# 冒烟测试 / 主入口
# ----------------------------------------------------------------------------
def smoke():
    """单次计时：BIAM / NAM / SAM"""
    set_seed(42)
    cfg = BIAMConfig(); cfg.task = 'regression'; cfg.seed = 42
    cfg.noise_ratio = 0.3; cfg.missing_ratio = 0.2
    gen = BIAMDataGenerator(cfg)
    _, _, train_data, val_data, test_data = gen.generate_data()
    t0 = time.time()
    m = run_biam('regression', train_data, val_data, test_data, 42, epochs=20, variant='BIAM')
    dt = time.time() - t0
    print(f"[smoke] BIAM 20 epochs: {dt:.1f}s -> 200 epochs est {dt*10:.0f}s, metrics={m}")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data
    Xtr_i, Xte_i = impute_mean(X_train, X_test)
    t0 = time.time()
    net = NAMNet(X_train.shape[1], task='regression').to(DEVICE)
    net = train_torch_baseline(net, Xtr_i, y_train, 'regression', 42, epochs=300)
    print(f"[smoke] NAM 300 epochs: {time.time()-t0:.1f}s, mse={reg_metrics(y_test, predict_torch(net, Xte_i, 'regression'))}")
    t0 = time.time()
    pred = run_sam('regression', Xtr_i, y_train, Xte_i, 42)
    print(f"[smoke] SAM: {time.time()-t0:.1f}s, mse={reg_metrics(y_test, pred)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', nargs='+', default=['E1', 'E2', 'E3'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--biam2-only', action='store_true',
                        help='仅重跑 BIAM2 行（跳过基线）')
    args = parser.parse_args()
    if args.biam2_only:
        BIAM2_ONLY = True

    print(f"Device: {DEVICE} | seeds: {args.seeds}", flush=True)
    if args.smoke:
        smoke()
        sys.exit(0)

    os.makedirs(EXP_DIR, exist_ok=True)
    for block in args.blocks:
        print(f"\n===== Block {block} start =====", flush=True)
        t0 = time.time()
        {'E1': block_E1, 'E2': block_E2, 'E3': block_E3}[block](args.seeds)
        print(f"===== Block {block} done in {(time.time()-t0)/60:.1f} min =====", flush=True)
