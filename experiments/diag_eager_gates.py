"""
诊断实验：验证"快门控 + 强 ℓ0 惩罚"假设
============================================================================
背景：第 2 轮校准（warmup=10 + gate_lr_scale=0.3 + λ0=0.05）修复了
E1/outlier30 的门控失控，但 5 种子全量 E6 显示该保守机制使 BIAM2 在
交互场景全面退化（int_clean 0.789 vs 旧机制 0.459）。

假设：E1/outlier30 失控的根因是旧机制 λ0=1e-4 过弱，而非门控速度本身。
若"无预热 + 全速门控 + λ0=0.05"能同时满足：
  D1 E1_outlier30 : mse ≤ ~0.8（BIAM=1.574）
  D2 E1_mixture_A : mse 不恶化（BIAM=0.578）
  D3 E6_int_clean : mse ≤ ~0.5 且 AUPRC ≥ 0.7
  D4 E6_null_main : AUPRC ≈ 随机（≤0.1）
则采用该更简单的机制（仅独立 λ_pair 正则，去掉预热与慢门控）。

用法：python experiments/diag_eager_gates.py
结果追加写入 exp/calibration_pair_reg.csv（tag: BIAM2_eager_l0=0.05）
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import time
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_new_experiments import (
    set_seed, run_biam, BIAMDataGenerator, apply_regression_noise,
)
from run_biam2_experiments import (
    gen_interact_regression, run_biam2_full, pair_auprc, TRUE_PAIRS,
)
from utils.biam_config import BIAMConfig

SEED = 42
PAIR_L0, PAIR_L2 = 0.05, 0.01
WARMUP, GATE_SCALE = 0, 1.0     # 快门控：无预热、门控与主效应同学习率
TAG = f'BIAM2_eager_w{WARMUP}_s{GATE_SCALE}_l0={PAIR_L0}'
OUT_CSV = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp', 'calibration_pair_reg.csv')
COLS = ['block', 'scenario', 'method', 'seed', 'time_sec', 'auprc', 'mse', 'rmse', 'mae']


def append_cal(row: dict):
    row = {c: row.get(c, float('nan')) for c in COLS}
    header = not os.path.exists(OUT_CSV)
    pd.DataFrame([row], columns=COLS).to_csv(OUT_CSV, mode='a', header=header, index=False)
    print(f"    [diag] {row['scenario']} {row['method']} "
          f"metrics={ {k: (round(v, 4) if isinstance(v, float) and v == v else v) for k, v in row.items() if k in ('mse', 'auprc')} }",
          flush=True)


def gen_e1_scenario(scenario, seed):
    """E1 协议：无缺失、无内置噪声，训练标签注入指定腐蚀"""
    set_seed(seed)
    cfg = BIAMConfig(); cfg.task = 'regression'; cfg.seed = seed
    cfg.noise_ratio = 0; cfg.missing_ratio = 0
    gen = BIAMDataGenerator(cfg)
    _, _, train_data, val_data, test_data = gen.generate_data()
    (X_train, y_train), _, _ = train_data, val_data, test_data
    y_noisy = apply_regression_noise(scenario, y_train, seed)
    return (X_train, y_noisy), val_data, test_data


def main():
    # 断点续跑
    done = set()
    if os.path.exists(OUT_CSV):
        done = set(zip(pd.read_csv(OUT_CSV)['scenario'], pd.read_csv(OUT_CSV)['method']))

    # D1/D2: E1 腐蚀场景
    for scenario in ('outlier30', 'mixture_A'):
        if (f'E1_{scenario}', TAG) in done:
            print(f'[skip] E1_{scenario}', flush=True)
            continue
        train_noisy, val_data, test_data = gen_e1_scenario(scenario, SEED)
        t0 = time.time()
        m = run_biam('regression', train_noisy, val_data, test_data, SEED,
                     variant='BIAM2', pair_l0=PAIR_L0, pair_l2=PAIR_L2,
                     pair_warmup=WARMUP, gate_lr_scale=GATE_SCALE)
        append_cal({'block': 'CAL', 'scenario': f'E1_{scenario}', 'method': TAG,
                    'seed': SEED, 'time_sec': round(time.time() - t0, 1), **m})

    # D3/D4: E6 交互场景
    for scenario in ('int_clean', 'null_main'):
        if (f'E6_{scenario}', TAG) in done:
            print(f'[skip] E6_{scenario}', flush=True)
            continue
        (train_data, val_data, test_data), true_pairs = gen_interact_regression(scenario, SEED)
        t0 = time.time()
        m, S = run_biam2_full('regression', train_data, val_data, test_data, SEED,
                              pair_l0=PAIR_L0, pair_l2=PAIR_L2,
                              pair_warmup=WARMUP, gate_lr_scale=GATE_SCALE)
        append_cal({'block': 'CAL', 'scenario': f'E6_{scenario}', 'method': TAG,
                    'seed': SEED, 'time_sec': round(time.time() - t0, 1),
                    'auprc': pair_auprc(S, TRUE_PAIRS), **m})

    print('诊断完成', flush=True)


if __name__ == '__main__':
    main()
