"""
剔除结果 CSV 中的 BIAM2 行（先整体备份到 exp/backup_pre_pair_reg/），
用于成对门控独立正则（lambda_pair_l2/l0）修复后的增量重跑。

用法：
  python experiments/strip_biam2_rows.py            # 处理全部 E1-E7
  python experiments/strip_biam2_rows.py E1 E4      # 仅处理指定块
"""
import os
import sys
import shutil
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp')
BACKUP_DIR = os.path.join(EXP_DIR, 'backup_pre_pair_reg')

FILES = {
    'E1': 'results_new_E1.csv', 'E2': 'results_new_E2.csv', 'E3': 'results_new_E3.csv',
    'E4': 'results_new_E4.csv', 'E5': 'results_new_E5.csv',
    'E6': 'results_biam2_E6.csv', 'E7': 'results_biam2_E7.csv',
}


def main():
    blocks = sys.argv[1:] or list(FILES)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for blk in blocks:
        path = os.path.join(EXP_DIR, FILES[blk])
        if not os.path.exists(path):
            print(f'[skip] {FILES[blk]} 不存在')
            continue
        shutil.copy2(path, os.path.join(BACKUP_DIR, FILES[blk]))
        df = pd.read_csv(path)
        n0 = len(df)
        df = df[df.method != 'BIAM2']
        df.to_csv(path, index=False)
        print(f'[{blk}] {n0} -> {len(df)} 行（BIAM2 行已剔除，原文件备份至 backup_pre_pair_reg/）')


if __name__ == '__main__':
    main()
