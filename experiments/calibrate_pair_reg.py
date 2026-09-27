"""
BIAM² 成对门控正则系数（lambda_pair_l0 / lambda_pair_l2）校准脚本

目的：标签腐蚀（离群值）场景下，全局弱 ℓ0 正则不足以关闭成对门控，
导致二阶容量拟合离群标签、MSE 爆炸。本脚本在三组代表性场景上扫描
λ_pair_l0（λ_pair_l2 固定），确定最终默认值：
  - E1/outlier30 : 30% 大幅离群标签 → 要求 mse ≤ BIAM（≈1.57）
  - E6/int_clean : 植入交互 + 无噪声 → 要求 AUPRC ≥ 0.7 且 mse ≤ 0.4
  - E6/null_main : 无交互对照 + 30% 噪声 → 要求 AUPRC ≈ 随机(≈0.074)

用法：
  python experiments/calibrate_pair_reg.py
结果输出： exp/calibration_pair_reg.csv
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
    DEVICE, set_seed, run_biam,
    BIAMDataGenerator, apply_regression_noise,
)
from run_biam2_experiments import (
    gen_interact_regression, run_biam2_full, pair_auprc, TRUE_PAIRS,
)
from utils.biam_config import BIAMConfig

SEED = 42
PAIR_L2 = 0.01
# 第 2 轮校准：warmup 10 + 门控学习率 0.3（第 1 轮证明 λ0 不敏感的根因是
# ℓ0 替代项在 γ=0 处梯度为 0，开门初期毫无约束力；此处放大门控步长救
# int_clean 的 mse，同时用更大的 λ0 检验 γ>0 后能否把 outlier30 的门控拉回）
CONFIGS = [
    # (pair_warmup, gate_lr_scale, pair_l0)
    (10, 0.3, 0.05),
    (10, 0.3, 0.2),
]
OUT_CSV = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp', 'calibration_pair_reg.csv')


COLS = ['block', 'scenario', 'method', 'seed', 'time_sec', 'auprc', 'mse', 'rmse', 'mae']


def append_cal(row: dict):
    """校准结果独立落盘（固定列序，防止不同场景列集不一致导致 CSV 错位）"""
    row = {c: row.get(c, float('nan')) for c in COLS}
    header = not os.path.exists(OUT_CSV)
    pd.DataFrame([row], columns=COLS).to_csv(OUT_CSV, mode='a', header=header, index=False)
    print(f"    [cal] {row['scenario']} {row['method']} seed={row['seed']} "
          f"metrics={ {k: (round(v, 4) if isinstance(v, float) and v == v else v) for k, v in row.items() if k in ('mse', 'mae', 'auprc')} }",
          flush=True)


def gen_e1_outlier30(seed):
    """E1 协议：无缺失、无内置噪声，训练标签注入 30% 大幅离群值"""
    set_seed(seed)
    cfg = BIAMConfig(); cfg.task = 'regression'; cfg.seed = seed
    cfg.noise_ratio = 0; cfg.missing_ratio = 0
    gen = BIAMDataGenerator(cfg)
    _, _, train_data, val_data, test_data = gen.generate_data()
    (X_train, y_train), _, _ = train_data, val_data, test_data
    y_train_noisy = apply_regression_noise('outlier30', y_train, seed)
    return (X_train, y_train_noisy), val_data, test_data


def main():
    # 断点续跑：读取已有校准结果，跳过已完成的 (场景, 配置) 组合
    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev['scenario'], prev['method']))
        print(f'续跑检测：已有 {len(done)} 条完成记录，将跳过对应组合', flush=True)

    for warmup, scale, pair_l0 in CONFIGS:
        tag = f'BIAM2_w{warmup}_s{scale}_l0={pair_l0}'
        plan = [('E1_outlier30',)] + [(f'E6_{s}',) for s in ('int_clean', 'null_main')]
        todo = [p[0] for p in plan if (p[0], tag) not in done]
        if not todo:
            print(f'[skip] {tag} 全部完成', flush=True)
            continue

        # --- E1/outlier30：腐蚀场景下不得拟合离群标签 ---
        if 'E1_outlier30' in todo:
            train_noisy, val_data, test_data = gen_e1_outlier30(SEED)
            t0 = time.time()
            m = run_biam('regression', train_noisy, val_data, test_data, SEED,
                         variant='BIAM2', pair_l0=pair_l0, pair_l2=PAIR_L2,
                         pair_warmup=warmup, gate_lr_scale=scale)
            append_cal({'block': 'CAL', 'scenario': 'E1_outlier30', 'method': tag,
                        'seed': SEED, 'time_sec': round(time.time() - t0, 1), **m})

        # --- E6/int_clean 与 E6/null_main：交互恢复能力不得被正则破坏 ---
        for scenario in ('int_clean', 'null_main'):
            if f'E6_{scenario}' not in todo:
                continue
            (train_data, val_data, test_data), true_pairs = gen_interact_regression(scenario, SEED)
            t0 = time.time()
            m, S = run_biam2_full('regression', train_data, val_data, test_data, SEED,
                                  pair_l0=pair_l0, pair_l2=PAIR_L2,
                                  pair_warmup=warmup, gate_lr_scale=scale)
            auprc = pair_auprc(S, TRUE_PAIRS)
            append_cal({'block': 'CAL', 'scenario': f'E6_{scenario}', 'method': tag,
                        'seed': SEED, 'time_sec': round(time.time() - t0, 1),
                        'auprc': auprc, **m})

    print('校准完成，结果见', os.path.abspath(OUT_CSV), flush=True)


if __name__ == '__main__':
    main()
