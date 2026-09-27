"""
补齐 E7/clf_null_main 的 5 个种子 BIAM2 行（发现档门控调度）。

发现档 = warmup 0 + 全速门控 + 门控正则关闭（与 backup_pre_pair_reg 中
E6/E7 旧 BIAM2 行的机制一致），用于 §7 植入交互基准。
结果追加写入 exp/results_biam2_E7.csv（断点续跑：跳过已有行）。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_new_experiments import DEVICE, set_seed
from run_biam2_experiments import (
    gen_interact_classification, run_biam2_full, pair_auprc,
    TRUE_PAIRS_CLF, result_csv, append_row,
)

SEEDS = [42, 7, 123, 2024, 3407]
SCENARIO = 'clf_null_main'


def main():
    path = result_csv('E7')
    done = set()
    if os.path.exists(path):
        df = pd.read_csv(path)
        done = set(zip(df.scenario, df.method, df.seed))

    for seed in SEEDS:
        if (SCENARIO, 'BIAM2', seed) in done:
            print(f'[skip] {SCENARIO} BIAM2 seed={seed}', flush=True)
            continue
        (train_data, val_data, test_data), true_pairs = gen_interact_classification(SCENARIO, seed)
        t0 = time.time()
        # 发现档：无预热、全速门控、门控正则关闭
        m, S = run_biam2_full('classification', train_data, val_data, test_data, seed,
                              pair_l0=0.0, pair_l2=0.0, pair_warmup=0, gate_lr_scale=1.0)
        append_row({'block': 'E7', 'scenario': SCENARIO, 'method': 'BIAM2', 'seed': seed,
                    'time_sec': round(time.time() - t0, 1),
                    'auprc_pair': pair_auprc(S, TRUE_PAIRS_CLF) if S is not None else np.nan, **m})

    print('clf_null_main BIAM2 补跑完成', flush=True)


if __name__ == '__main__':
    main()
