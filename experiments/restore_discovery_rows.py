"""
恢复 E6/E7 的"发现档"（旧机制）BIAM2 行，并归档新配置（鲁棒档）行。

背景：双档门控调度方案——
  鲁棒档 (warmup=10, gate_lr_scale=0.3, λ_pair=0.05/0.01)：E1-E5 加性压力测试
  发现档 (warmup=0, gate_lr_scale=1.0, 门控正则≈0)：E6/E7 植入交互基准
backup_pre_pair_reg/ 中的 E6/E7 BIAM2 行即发现档结果（与 §7 协议一致），
当前 CSV 中的 BIAM2 行为鲁棒档产出，归档至 backup_robust_e6e7/ 后替换。
"""
import os
import shutil
import pandas as pd

EXP = r'c:\Users\Administrator\Desktop\BIAM修改\exp'
BAK = os.path.join(EXP, 'backup_pre_pair_reg')
ARCHIVE = os.path.join(EXP, 'backup_robust_e6e7')

for blk, metric in [('E6', None), ('E7', None)]:
    fname = f'results_biam2_{blk}.csv'
    cur = pd.read_csv(os.path.join(EXP, fname))
    old = pd.read_csv(os.path.join(BAK, fname))

    cur_b2 = cur[cur.method == 'BIAM2']
    old_b2 = old[old.method == 'BIAM2']

    # 归档当前鲁棒档行
    os.makedirs(ARCHIVE, exist_ok=True)
    cur_b2.to_csv(os.path.join(ARCHIVE, fname), index=False)

    # 当前 CSV 去掉 BIAM2 行，接回发现档行
    rest = cur[cur.method != 'BIAM2']
    merged = pd.concat([rest, old_b2], ignore_index=True)
    merged.to_csv(os.path.join(EXP, fname), index=False)
    print(f'{fname}: 保留非BIAM2 {len(rest)} 行 + 恢复发现档 BIAM2 {len(old_b2)} 行'
          f'（鲁棒档 {len(cur_b2)} 行已归档至 backup_robust_e6e7/）')
