"""
E4B：补齐 E4 三个制表场景（laplace / outlier50 / outlier70）基线方法的 MAE。
当前 crc32 协议重跑时基线行未含 MAE，论文 tab:extreme 的 MAE 行需要。
结果写入独立的 exp/results_new_E4B.csv（不混入主结果，避免重复行）。
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
    SEEDS, set_seed, impute_mean, run_sam, run_xgb, reg_metrics,
    NAMNet, MLPNet, train_torch_baseline, predict_torch, DEVICE,
)
from run_extension_experiments import apply_ext_noise, run_huber, run_ransac
from run_new_experiments import run_lasso
from data.biam_data_generator import BIAMDataGenerator
from utils.biam_config import BIAMConfig

EXP_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'exp')
OUT = os.path.join(EXP_DIR, 'results_new_E4B.csv')
SCENARIOS = ['laplace', 'outlier50', 'outlier70']


def main():
    done = set()
    if os.path.exists(OUT):
        df = pd.read_csv(OUT)
        done = set(zip(df.scenario, df.method, df.seed))

    for scenario in SCENARIOS:
        for seed in SEEDS:
            set_seed(seed)
            cfg = BIAMConfig(); cfg.task = 'regression'; cfg.seed = seed
            cfg.noise_ratio = 0; cfg.missing_ratio = 0
            gen = BIAMDataGenerator(cfg)
            _, _, train_data, val_data, test_data = gen.generate_data()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data
            y_noisy = apply_ext_noise(scenario, y_train, seed)

            Xtr_i, Xte_i = impute_mean(X_train, X_test)
            preds = {
                'Lasso': run_lasso('regression', Xtr_i, y_noisy, Xte_i, seed),
                'SAM': run_sam('regression', Xtr_i, y_noisy, Xte_i, seed),
                'Huber': run_huber(Xtr_i, y_noisy, Xte_i),
                'RANSAC': run_ransac(Xtr_i, y_noisy, Xte_i, seed),
                'XGBoost': run_xgb('regression', Xtr_i, y_noisy, Xte_i, seed),
            }
            for name, net in (('NAM', NAMNet(X_train.shape[1], task='regression')),
                              ('MLP', MLPNet(X_train.shape[1], task='regression'))):
                net = net.to(DEVICE)
                net = train_torch_baseline(net, Xtr_i, y_noisy, 'regression', seed)
                preds[name] = predict_torch(net, Xte_i, 'regression')

            for name, pred in preds.items():
                if (scenario, name, seed) in done:
                    continue
                row = {'block': 'E4B', 'scenario': scenario, 'method': name, 'seed': seed,
                       'time_sec': 0.0, **reg_metrics(y_test, pred)}
                header = not os.path.exists(OUT)
                pd.DataFrame([row]).to_csv(OUT, mode='a', header=header, index=False)
                print(f"[saved] E4B/{scenario}/{name} seed={seed} "
                      f"mae={row['mae']:.4f}", flush=True)

    print('E4B 完成', flush=True)


if __name__ == '__main__':
    main()
