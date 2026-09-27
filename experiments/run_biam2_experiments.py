"""
BIAM² 成对交互实验（参考 KNOSYS-D-26-21566 / AG-NAM 的交互发现协议）
============================================================================
设计思想（AG-NAM）：在合成数据中植入已知的成对交互（乘积 / sin 乘积 / tanh 乘积），
检验模型能否 (a) 预测得更好，(b) 通过成对交互强度矩阵正确"排名"出真实交互对
（interaction-ranking AUPRC），并设置无交互对照（null）场景。

E6 回归交互谱系（n=1200, p=10，三个真实交互对）：
   - int_mcar / int_mar / int_mnar（交互 + 20% 缺失，30% 标签噪声）
   - int_clean（交互、无缺失、无噪声）
   - null_main（无交互对照：仅主效应 + 30% 标签噪声 + 20% MCAR）
E7 分类交互谱系（logistic 主效应 + 两个交互对）：
   - clf_int_mcar / clf_int_mar / clf_int_mnar / clf_int_clean / clf_null_main

对比方法：BIAM2（本文二阶扩展）、BIAM、Lasso、SAM、NAM、MLP、XGBoost、
          EBM（Explainable Boosting Machine, InterpretML）、CatBoost
          （EBM/CatBoost 均为开源实现且可在 Google Scholar 检索到对应论文）
指标：回归 MSE/MAE；分类 ACC/Macro-F1；交互排序 AUPRC（BIAM2 专属输出）

用法：
  python experiments/run_biam2_experiments.py --blocks E6 E7
  python experiments/run_biam2_experiments.py --smoke
结果输出： exp/results_biam2_E6.csv / results_biam2_E7.csv
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import argparse
import time
import zlib
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 复用主实验脚本的通用组件
from run_new_experiments import (
    SEEDS, DEVICE, set_seed, impute_mean, run_biam,
    NAMNet, MLPNet, train_torch_baseline, predict_torch,
    run_lasso, run_sam, run_xgb, reg_metrics, clf_metrics,
)

from utils.biam_config import BIAMConfig
from models.biam_model import BIAMModel
from models.biam_weighting_network import BIAMWeightingNetwork
from gradients.biam_optimizer import BIAMOptimizer
from sklearn.metrics import average_precision_score

EXP_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp')

# --biam2-only：仅重跑 BIAM2 行（跳过 BIAM 对照与基线），用于正则修复后的增量重跑
BIAM2_ONLY = False


def result_csv(block: str) -> str:
    return os.path.join(EXP_DIR, f'results_biam2_{block}.csv')


def append_row(row: dict):
    df = pd.DataFrame([row])
    path = result_csv(row['block'])
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)
    print(f"    [saved] {row['block']}/{row['scenario']}/{row['method']} seed={row['seed']} "
          f"metrics={ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in ('mse', 'mae', 'accuracy', 'f1_macro', 'auprc_pair')} }",
          flush=True)


def det_rng(tag: str, seed: int) -> np.random.Generator:
    """跨进程确定性的随机源（hash() 在 Windows 下逐进程随机化，不可用）"""
    return np.random.default_rng(zlib.crc32(f'{tag}-{seed}'.encode()) + seed)


# ----------------------------------------------------------------------------
# 植入成对交互的数据生成（AG-NAM S1-S4 思想，规模与论文协议对齐）
# ----------------------------------------------------------------------------
P = 10
TRUE_PAIRS = [(4, 5), (6, 7), (8, 9)]        # 回归三个交互对
TRUE_PAIRS_CLF = [(4, 5), (6, 7)]            # 分类两个交互对


def gen_interact_regression(scenario: str, seed: int):
    """回归：y = 主效应 + 植入交互 + (30% 标签噪声) + (20% 缺失)
    X、y 先按训练集统计量标准化（与论文协议一致），噪声与缺失作用于标准化尺度"""
    rng = det_rng('e6-gen', seed)
    n = 1200
    X = rng.normal(size=(n, P)).astype(np.float32)
    main = 1.5 * np.sin(X[:, 0]) + (X[:, 1] ** 2 - 1.0) - 1.2 * X[:, 3]
    inter = (2.0 * X[:, 4] * X[:, 5]
             + 1.5 * np.sin(X[:, 6] * X[:, 7])
             + 1.2 * np.tanh(X[:, 8]) * np.tanh(X[:, 9]))
    if scenario == 'null_main':                  # 无交互对照
        inter = np.zeros(n)
    y = (main + inter + 0.1 * rng.normal(size=n)).astype(np.float32)

    # 划分索引 → 用训练集统计量标准化（不引入测试集信息）
    idx = rng.permutation(n)
    tr, va, te = idx[:840], idx[840:960], idx[960:]
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-8
    X = (X - mu_x) / sd_x
    mu_y, sd_y = y[tr].mean(), y[tr].std() + 1e-8
    y = (y - mu_y) / sd_y

    # 30% 标签噪声仅作用于训练集标签（E1 协议）
    if scenario != 'int_clean':
        noisy = rng.random(len(tr)) < 0.3
        y[tr] += np.where(noisy, rng.normal(1.0, 0.5, len(tr)), 0.0).astype(np.float32)

    # 20% 缺失（MAR 依赖 x0，MNAR 依赖自身）
    if scenario in ('int_mcar', 'int_mar', 'int_mnar', 'null_main'):
        if scenario == 'int_mcar':
            mask = rng.random(X.shape) < 0.2
        elif scenario == 'int_mar':
            probs = 1.0 / (1.0 + np.exp(-X[:, 0]))
            mask = rng.random(X.shape) < (probs * 0.4)[:, None]
        else:  # int_mnar
            probs = 1.0 / (1.0 + np.exp(-np.abs(X)))
            mask = rng.random(X.shape) < (probs * 0.4)
        mask[:, 0] = False                        # MAR/MNAR 的机制变量保持观测
        X = X.copy()
        X[mask] = np.nan

    y = y.reshape(-1, 1)
    return ((X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])), TRUE_PAIRS


def gen_interact_classification(scenario: str, seed: int):
    """分类：logistic(主效应 + 植入交互)；20% 噪声标签翻转；20% 缺失
    X 按训练集统计量标准化"""
    rng = det_rng('e7-gen', seed)
    n = 1200
    X = rng.normal(size=(n, P)).astype(np.float32)
    logit = (np.sin(X[:, 0]) + 0.8 * np.tanh(X[:, 2]) - 0.9 * X[:, 3]
             + 1.6 * X[:, 4] * X[:, 5] + 1.3 * np.sin(X[:, 6] * X[:, 7]))
    if scenario == 'clf_null_main':
        logit -= (1.6 * X[:, 4] * X[:, 5] + 1.3 * np.sin(X[:, 6] * X[:, 7]))
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(np.int64)

    # 划分索引 → 用训练集统计量标准化 X
    idx = rng.permutation(n)
    tr, va, te = idx[:840], idx[840:960], idx[960:]
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-8
    X = (X - mu_x) / sd_x

    if scenario != 'clf_int_clean':              # 20% 标签翻转仅作用训练集
        flip = rng.random(len(tr)) < 0.2
        y[tr[flip]] = 1 - y[tr[flip]]

    if scenario in ('clf_int_mcar', 'clf_int_mar', 'clf_int_mnar', 'clf_null_main'):
        if scenario == 'clf_int_mcar':
            mask = rng.random(X.shape) < 0.2
        elif scenario == 'clf_int_mar':
            probs = 1.0 / (1.0 + np.exp(-X[:, 0]))
            mask = rng.random(X.shape) < (probs * 0.4)[:, None]
        else:
            probs = 1.0 / (1.0 + np.exp(-np.abs(X)))
            mask = rng.random(X.shape) < (probs * 0.4)
        mask[:, 0] = False
        X = X.copy()
        X[mask] = np.nan

    return ((X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])), TRUE_PAIRS_CLF


# ----------------------------------------------------------------------------
# BIAM2 / BIAM
# ----------------------------------------------------------------------------
def run_biam2_full(task, train_data, val_data, test_data, seed,
                   epochs=None, batch_size=64, pair_l0=None, pair_l2=None,
                   pair_warmup=None, gate_lr_scale=None):
    """训练 BIAM2 并返回 (metrics, 成对交互强度矩阵 (p,p))
    校准配置：秩 R=8；回归 300 epochs / 分类 200 epochs"""
    if epochs is None:
        epochs = 300 if task == 'regression' else 200
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

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data
    config.input_dim = X_train.shape[1]
    if task == 'classification':
        config.num_classes = 2

    import torch.utils.data as Data
    yt = y_train if task == 'classification' else y_train.reshape(-1, 1)
    yv = y_val if task == 'classification' else y_val.reshape(-1, 1)
    train_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                           torch.tensor(np.asarray(yt), dtype=torch.float32)),
        batch_size=batch_size, shuffle=True)
    val_loader = Data.DataLoader(
        Data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(np.asarray(yv), dtype=torch.float32)),
        batch_size=batch_size, shuffle=False)

    model = BIAMModel(config, DEVICE)
    weighting = BIAMWeightingNetwork(config, DEVICE)
    model.additive_model.set_knots(torch.tensor(X_train, dtype=torch.float32))
    optimizer = BIAMOptimizer(config, model, weighting)
    for epoch in range(epochs):
        optimizer.train_epoch(train_loader, val_loader, epoch)

    metrics = optimizer.evaluate(test_data)
    S = model.additive_model.get_pair_strengths()
    return metrics, S


def pair_auprc(S: np.ndarray, true_pairs) -> float:
    """成对交互排序 AUPRC：真实交互对为正类，C(p,2) 个无序对为候选"""
    p = S.shape[0]
    scores, labels = [], []
    truth = set(map(tuple, true_pairs))
    for j in range(p):
        for k in range(j + 1, p):
            scores.append(S[j, k])
            labels.append(1.0 if (j, k) in truth else 0.0)
    if sum(labels) == 0:
        return float('nan')
    return float(average_precision_score(labels, scores))


# ----------------------------------------------------------------------------
# EBM / CatBoost（新增对比方法，均开源且可检索）
# ----------------------------------------------------------------------------
def run_ebm(task, X_train, y_train, X_test, seed):
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    set_seed(seed)
    # EBM 不接受 NaN → 均值填充（与其他基线协议一致）
    mu = np.nanmean(X_train, axis=0)
    Xtr = np.where(np.isnan(X_train), mu, X_train)
    Xte = np.where(np.isnan(X_test), mu, X_test)
    if task == 'regression':
        model = ExplainableBoostingRegressor(random_state=seed)
        model.fit(Xtr, y_train)
        return model.predict(Xte)
    model = ExplainableBoostingClassifier(random_state=seed)
    model.fit(Xtr, np.asarray(y_train).astype(int))
    return model.predict(Xte)


def run_catboost(task, X_train, y_train, X_test, seed):
    from catboost import CatBoostRegressor, CatBoostClassifier
    set_seed(seed)
    common = dict(iterations=300, depth=4, learning_rate=0.1, random_seed=seed,
                  verbose=False, allow_writing_files=False)
    if task == 'regression':
        model = CatBoostRegressor(**common)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    model = CatBoostClassifier(**common)
    model.fit(X_train, np.asarray(y_train).astype(int))
    return model.predict(X_test).astype(int)


# ----------------------------------------------------------------------------
# 实验块
# ----------------------------------------------------------------------------
def _existing_rows(block):
    """已完成的 (scenario, method, seed) 集合，用于断点续跑"""
    path = result_csv(block)
    done = set()
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        done = set(zip(df.scenario, df.method, df.seed))
    return done


def run_block(block: str, seeds):
    task = 'regression' if block == 'E6' else 'classification'
    gen = gen_interact_regression if block == 'E6' else gen_interact_classification
    if block == 'E6':
        scenarios = ['int_clean', 'int_mcar', 'int_mar', 'int_mnar', 'null_main']
    else:
        scenarios = ['clf_int_clean', 'clf_int_mcar', 'clf_int_mar',
                     'clf_int_mnar', 'clf_null_main']

    done = _existing_rows(block)
    for scenario in scenarios:
        for seed in seeds:
            (train_data, val_data, test_data), true_pairs = gen(scenario, seed)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data

            # --- BIAM2（含成对交互强度） ---
            if (scenario, 'BIAM2', seed) not in done:
                t0 = time.time()
                m, S = run_biam2_full(task, train_data, val_data, test_data, seed)
                row = {'block': block, 'scenario': scenario, 'method': 'BIAM2', 'seed': seed,
                       'time_sec': round(time.time() - t0, 1),
                       'auprc_pair': pair_auprc(S, true_pairs) if S is not None else np.nan, **m}
                append_row(row)

            if BIAM2_ONLY:
                continue

            # --- BIAM（一阶对照） ---
            if (scenario, 'BIAM', seed) not in done:
                t0 = time.time()
                m = run_biam(task, train_data, val_data, test_data, seed)
                append_row({'block': block, 'scenario': scenario, 'method': 'BIAM', 'seed': seed,
                            'time_sec': round(time.time() - t0, 1), 'auprc_pair': np.nan, **m})

            # --- 基线（均值填充，CatBoost 原生支持缺失） ---
            Xtr_i, Xte_i = impute_mean(X_train, X_test)
            BASELINES = ['Lasso', 'SAM', 'XGBoost', 'EBM', 'CatBoost', 'NAM', 'MLP']
            todo = [b for b in BASELINES if (scenario, b, seed) not in done]
            if not todo:
                continue
            preds = {
                'Lasso': run_lasso(task, Xtr_i, y_train, Xte_i, seed),
                'SAM': run_sam(task, Xtr_i, y_train, Xte_i, seed),
                'XGBoost': run_xgb(task, Xtr_i, y_train, Xte_i, seed),
                'EBM': run_ebm(task, X_train, y_train, X_test, seed),
                'CatBoost': run_catboost(task, X_train, y_train, X_test, seed),
            }
            for name, net in (('NAM', NAMNet(X_train.shape[1], task=task)),
                              ('MLP', MLPNet(X_train.shape[1], task=task))):
                net = net.to(DEVICE)
                net = train_torch_baseline(net, Xtr_i, y_train, task, seed)
                preds[name] = predict_torch(net, Xte_i, task)

            for name, pred in preds.items():
                if name not in todo:
                    continue
                mm = reg_metrics(y_test, pred) if task == 'regression' else clf_metrics(y_test, pred)
                append_row({'block': block, 'scenario': scenario, 'method': name, 'seed': seed,
                            'time_sec': np.nan, 'auprc_pair': np.nan, **mm})


def smoke():
    """单场景单种子快速计时"""
    t0 = time.time()
    (tr, va, te), pairs = gen_interact_regression('int_mcar', 42)
    m, S = run_biam2_full('regression', tr, va, te, 42, epochs=20)
    print(f"[smoke] BIAM2 20ep: {time.time()-t0:.1f}s, mse={m['mse']:.4f}, "
          f"auprc={pair_auprc(S, pairs):.3f}")
    t0 = time.time()
    pred = run_ebm('regression', tr[0], tr[1], te[0], 42)
    print(f"[smoke] EBM: {time.time()-t0:.1f}s, mse={reg_metrics(te[1], pred)['mse']:.4f}")
    t0 = time.time()
    pred = run_catboost('regression', tr[0], tr[1], te[0], 42)
    print(f"[smoke] CatBoost: {time.time()-t0:.1f}s, mse={reg_metrics(te[1], pred)['mse']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', nargs='+', default=['E6', 'E7'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--biam2-only', action='store_true',
                        help='仅重跑 BIAM2 行（跳过 BIAM 对照与基线）')
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
        run_block(block, args.seeds)
        print(f"===== Block {block} done in {(time.time()-t0)/60:.1f} min =====", flush=True)
