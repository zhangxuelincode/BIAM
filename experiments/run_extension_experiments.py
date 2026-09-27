"""
扩展实验（对 0-PAPER.tex 现有实验设置的补充验证）：
============================================================================
E4 回归扩展：更多噪声/异常值类型 + 极端场景，指标同步 MSE+MAE（所有方法）
   - gauss / laplace / cauchy / lognormal / outlier50 / outlier70
   - smalln (n=300) / highdim (p=50)
   - 新增鲁棒线性基线：Huber、RANSAC
E5 分类扩展：极端类失衡 × 标签噪声 × 缺失三重场景，指标同步 ACC+Macro-F1
   - imb20_clean / imb50_clean / n20_imb20 / n30_imb50 / n20_imb50_miss20
   - 新增失衡感知基线：XGB-BAL（scale_pos_weight 加权 XGBoost）
E1B 指标同步：补齐 E1 中 Lasso/SAM/NAM/MLP/XGBoost 的 MAE（BIAM 家族已有）

用法：
  python experiments/run_extension_experiments.py --blocks E4 E5 E1B
结果输出： exp/results_new_E4.csv / E5 / E1B
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import argparse
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 复用主实验脚本的通用组件（该脚本带 __main__ 保护，可安全 import）
from run_new_experiments import (
    SEEDS, DEVICE, set_seed, impute_mean, append_row, run_biam, _existing_rows,
    NAMNet, MLPNet, train_torch_baseline, predict_torch,
    run_sam, run_xgb, reg_metrics, clf_metrics,
    E1_SCENARIOS, apply_regression_noise,
)

from data.biam_data_generator import BIAMDataGenerator
from utils.biam_config import BIAMConfig
from sklearn.linear_model import HuberRegressor, RANSACRegressor, LogisticRegressionCV
from sklearn.preprocessing import SplineTransformer

# --biam2-only：仅重跑 BIAM2 行（跳过基线），用于正则修复后的增量重跑
BIAM2_ONLY = False


# ----------------------------------------------------------------------------
# E4：回归扩展噪声与极端场景
# ----------------------------------------------------------------------------
def apply_ext_noise(scenario, y_train, seed):
    """与 E1 相同协议：噪声仅作用于标准化后的训练集 y
    用 zlib.crc32 而非 hash() 保证跨进程可复现（Windows 下 hash 逐进程随机化）"""
    import zlib
    rng = np.random.default_rng(2000 * seed + zlib.crc32(scenario.encode('utf-8')) % 9999)
    y = y_train.copy()
    n = len(y)
    if scenario == 'gauss':        # 普通加性高斯
        y += rng.normal(0, 1, n)
    elif scenario == 'laplace':    # 尖峰厚尾拉普拉斯
        y += rng.laplace(0, 1, n)
    elif scenario == 'cauchy':     # 极端重尾：柯西 t(1)
        y += rng.standard_t(df=1, size=n)
    elif scenario == 'lognormal':  # 偏态对数正态（中心化）
        y += rng.lognormal(0, 1, n) - np.exp(0.5)
    elif scenario in ('outlier50', 'outlier70'):
        r = 0.5 if scenario == 'outlier50' else 0.7
        idx = rng.choice(n, max(1, int(n * r)), replace=False)
        y[idx] += rng.normal(100, 100, len(idx))
    else:
        raise ValueError(scenario)
    return y


E4_SCENARIOS = ['gauss', 'laplace', 'cauchy', 'lognormal', 'outlier50', 'outlier70',
                'smalln', 'highdim']


def run_huber(X_train, y_train, X_test):
    model = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_ransac(X_train, y_train, X_test, seed):
    model = RANSACRegressor(random_state=seed, max_trials=200)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def block_E4(seeds):
    task = 'regression'
    done = _existing_rows('E4')
    for scenario in E4_SCENARIOS:
        for seed in seeds:
            set_seed(seed)
            cfg = BIAMConfig(); cfg.task = task; cfg.seed = seed
            if scenario == 'smalln':
                cfg.n_samples = 300
                cfg.noise_ratio = 0.3; cfg.noise_mean, cfg.noise_std = 1.0, 0.5
            elif scenario == 'highdim':
                cfg.n_samples = 1200; cfg.n_features = 50
                cfg.noise_ratio = 0.3; cfg.noise_mean, cfg.noise_std = 1.0, 0.5
            else:
                cfg.noise_ratio = 0; cfg.missing_ratio = 0
            cfg.missing_ratio = getattr(cfg, 'missing_ratio', 0)
            gen = BIAMDataGenerator(cfg)
            _, _, train_data, val_data, test_data = gen.generate_data()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data

            if scenario in ('smalln', 'highdim'):
                y_train_noisy = y_train  # 生成器内部已按 cfg.noise_ratio 注入
            else:
                y_train_noisy = apply_ext_noise(scenario, y_train, seed)
            train_noisy = (X_train, y_train_noisy)

            # --- BIAM 家族（GPU）---
            for variant in (('BIAM2',) if BIAM2_ONLY else ('BIAM2', 'BIAM', 'BIAM-B')):
                if (scenario, variant, seed) in done:
                    continue
                t0 = time.time()
                m = run_biam(task, train_noisy, val_data, test_data, seed, variant=variant)
                append_row({'block': 'E4', 'scenario': scenario, 'method': variant, 'seed': seed,
                            'time_sec': round(time.time() - t0, 1), **m})

            # --- 基线（均值填充）---
            if not BIAM2_ONLY:
                Xtr_i, Xte_i = impute_mean(X_train, X_test)
                preds = {
                    'Lasso': None, 'SAM': run_sam(task, Xtr_i, y_train_noisy, Xte_i, seed),
                    'Huber': run_huber(Xtr_i, y_train_noisy, Xte_i),
                    'RANSAC': run_ransac(Xtr_i, y_train_noisy, Xte_i, seed),
                    'XGBoost': run_xgb(task, Xtr_i, y_train_noisy, Xte_i, seed),
                }
                from run_new_experiments import run_lasso
                preds['Lasso'] = run_lasso(task, Xtr_i, y_train_noisy, Xte_i, seed)
                for name, net in (('NAM', NAMNet(X_train.shape[1], task=task)),
                                  ('MLP', MLPNet(X_train.shape[1], task=task))):
                    t0 = time.time()
                    net = net.to(DEVICE)
                    net = train_torch_baseline(net, Xtr_i, y_train_noisy, task, seed)
                    preds[name] = predict_torch(net, Xte_i, task)
                    m = reg_metrics(y_test, preds[name])
                    append_row({'block': 'E4', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': round(time.time() - t0, 1), **m})
                for name in ('Lasso', 'SAM', 'Huber', 'RANSAC', 'XGBoost'):
                    append_row({'block': 'E4', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': np.nan, **reg_metrics(y_test, preds[name])})


# ----------------------------------------------------------------------------
# E5：分类极端失衡 × 噪声 × 缺失
# ----------------------------------------------------------------------------
E5_SCENARIOS = [
    ('imb20_clean',  0.0, 0.05, 0.0),
    ('imb50_clean',  0.0, 0.02, 0.0),
    ('n20_imb20',    0.2, 0.05, 0.0),
    ('n30_imb50',    0.3, 0.02, 0.0),
    ('n20_imb50_miss20', 0.2, 0.02, 0.2),
]


def run_xgb_bal(task, X_train, y_train, X_test, seed):
    """失衡感知 XGBoost：scale_pos_weight 按类别频率加权"""
    from xgboost import XGBClassifier
    spw = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                          tree_method='hist', n_jobs=4, random_state=seed,
                          scale_pos_weight=spw, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model.predict(X_test)


def block_E5(seeds):
    task = 'classification'
    done = _existing_rows('E5')
    for scenario, r1, r2, r3 in E5_SCENARIOS:
        for seed in seeds:
            set_seed(seed)
            cfg = BIAMConfig(); cfg.task = task; cfg.seed = seed
            cfg.noise_ratio = r1
            cfg.imbalance_ratio = r2
            cfg.missing_ratio = r3
            if r3 > 0:
                cfg.missing_mechanism = 'MCAR'
            gen = BIAMDataGenerator(cfg)
            _, _, train_data, val_data, test_data = gen.generate_data()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data

            for variant in (('BIAM2',) if BIAM2_ONLY else ('BIAM2', 'BIAM', 'BIAM-B')):
                if (scenario, variant, seed) in done:
                    continue
                t0 = time.time()
                m = run_biam(task, train_data, val_data, test_data, seed, variant=variant)
                append_row({'block': 'E5', 'scenario': scenario, 'method': variant, 'seed': seed,
                            'time_sec': round(time.time() - t0, 1), **m})

            if not BIAM2_ONLY:
                Xtr_i, Xte_i = impute_mean(X_train, X_test)
                preds = {
                    'XGBoost': run_xgb(task, Xtr_i, y_train, Xte_i, seed),
                    'XGB-BAL': run_xgb_bal(task, Xtr_i, y_train, Xte_i, seed),
                }
                from run_new_experiments import run_lasso
                preds['Lasso'] = run_lasso(task, Xtr_i, y_train, Xte_i, seed)
                preds['SAM'] = run_sam(task, Xtr_i, y_train, Xte_i, seed)
                for name, net in (('NAM', NAMNet(X_train.shape[1], task=task)),
                                  ('MLP', MLPNet(X_train.shape[1], task=task))):
                    t0 = time.time()
                    net = net.to(DEVICE)
                    net = train_torch_baseline(net, Xtr_i, y_train, task, seed)
                    preds[name] = predict_torch(net, Xte_i, task)
                    m = clf_metrics(y_test, preds[name])
                    append_row({'block': 'E5', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': round(time.time() - t0, 1), **m})
                for name in ('Lasso', 'SAM', 'XGBoost', 'XGB-BAL'):
                    append_row({'block': 'E5', 'scenario': scenario, 'method': name, 'seed': seed,
                                'time_sec': np.nan, **clf_metrics(y_test, preds[name])})


# ----------------------------------------------------------------------------
# E1B：补齐 E1 基线的 MAE（指标同步），BIAM 家族已有 MAE 无需重跑
# ----------------------------------------------------------------------------
def block_E1B(seeds):
    task = 'regression'
    for scenario in E1_SCENARIOS:
        for seed in seeds:
            set_seed(seed)
            cfg = BIAMConfig(); cfg.task = task; cfg.seed = seed
            cfg.noise_ratio = 0; cfg.missing_ratio = 0
            gen = BIAMDataGenerator(cfg)
            _, _, train_data, val_data, test_data = gen.generate_data()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data
            y_train_noisy = apply_regression_noise(scenario, y_train, seed)

            Xtr_i, Xte_i = impute_mean(X_train, X_test)
            from run_new_experiments import run_lasso
            preds = {
                'Lasso': run_lasso(task, Xtr_i, y_train_noisy, Xte_i, seed),
                'SAM': run_sam(task, Xtr_i, y_train_noisy, Xte_i, seed),
                'XGBoost': run_xgb(task, Xtr_i, y_train_noisy, Xte_i, seed),
            }
            for name, net in (('NAM', NAMNet(X_train.shape[1], task=task)),
                              ('MLP', MLPNet(X_train.shape[1], task=task))):
                net = net.to(DEVICE)
                net = train_torch_baseline(net, Xtr_i, y_train_noisy, task, seed)
                preds[name] = predict_torch(net, Xte_i, task)
                append_row({'block': 'E1B', 'scenario': scenario, 'method': name, 'seed': seed,
                            'time_sec': np.nan, **reg_metrics(y_test, preds[name])})
            for name in ('Lasso', 'SAM', 'XGBoost'):
                append_row({'block': 'E1B', 'scenario': scenario, 'method': name, 'seed': seed,
                            'time_sec': np.nan, **reg_metrics(y_test, preds[name])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', nargs='+', default=['E4', 'E5', 'E1B'])
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--biam2-only', action='store_true',
                        help='仅重跑 BIAM2 行（跳过基线）')
    args = parser.parse_args()
    if args.biam2_only:
        BIAM2_ONLY = True

    print(f"Device: {DEVICE} | seeds: {args.seeds}", flush=True)
    os.makedirs(os.path.join(os.path.dirname(PROJECT_ROOT), 'exp'), exist_ok=True)
    for block in args.blocks:
        print(f"\n===== Block {block} start =====", flush=True)
        t0 = time.time()
        {'E4': block_E4, 'E5': block_E5, 'E1B': block_E1B}[block](args.seeds)
        print(f"===== Block {block} done in {(time.time()-t0)/60:.1f} min =====", flush=True)
