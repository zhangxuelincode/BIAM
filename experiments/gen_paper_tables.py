"""从当前 CSV 生成论文合成表的 LaTeX 行（crc32 可复现协议）
表1 tab:synthetic  : E1 MSE（8 方法列，含 BIAM2）
表2 tab:mae_syn    : E1 MAE（同上）
表3 tab:extreme    : E4 laplace/outlier50/outlier70 × MSE/MAE（9 方法列，无 BIAM2）
加粗规则：每行（每个指标）top-two。
另打印 §5 叙事所需的补充数字（cauchy/BIAM-B 对照等）与 E6/E7 汇总。
"""
import pandas as pd
import numpy as np

exp = r'c:\Users\Administrator\Desktop\BIAM修改\exp'


def fmt(mu, sd, bold):
    s = f'{mu:.3f} $\\pm$ {sd:.3f}'
    return f'\\textbf{{{s}}}' if bold else s


def top_two(means):
    order = means.sort_values()
    return set(order.index[:2])


def e1_table(metric, label):
    df = pd.read_csv(fr'{exp}\results_new_E1.csv')
    dfb = pd.read_csv(fr'{exp}\results_new_E1B.csv')
    cols = ['Lasso', 'SAM', 'NAM', 'MLP', 'XGBoost', 'BIAM-B', 'BIAM', 'BIAM2']
    scen_names = [('clean', 'None'), ('mixture_A', 'Gaussian mixture, shift $8$'),
                  ('mixture_B', 'Gaussian mixture, shift $20$'),
                  ('heavy_t', 'Heavy-tailed $t(2)$'),
                  ('outlier10', 'Outlier, $10\\%$'), ('outlier30', 'Outlier, $30\\%$')]
    print(f'===== {label} =====')
    for sc, disp in scen_names:
        g = df[df.scenario == sc]
        gb = dfb[dfb.scenario == sc]
        means, stds = {}, {}
        for m in cols:
            src = gb if (metric == 'mae' and m in gb.method.unique()) else g
            sel = src[src.method == m][metric]
            means[m], stds[m] = sel.mean(), sel.std()
        s = pd.Series(means)
        bolds = top_two(s)
        cells = ' & '.join(fmt(means[m], stds[m], m in bolds) for m in cols)
        print(f'{disp} & {cells} \\\\')
    print()


def e4_table():
    df = pd.read_csv(fr'{exp}\results_new_E4.csv')
    # 基线 MAE 来自 E4B 回填（BIAM 家族自身在 E4 主 CSV 中已有 MAE）
    dfb = pd.read_csv(fr'{exp}\results_new_E4B.csv')
    cols = ['Lasso', 'SAM', 'Huber', 'RANSAC', 'NAM', 'MLP', 'XGBoost', 'BIAM-B', 'BIAM']
    scen_names = [('laplace', 'Laplace'), ('outlier50', 'Outlier, $50\\%$'),
                  ('outlier70', 'Outlier, $70\\%$')]
    print('===== tab:extreme =====')
    for sc, disp in scen_names:
        g = df[df.scenario == sc]
        gb = dfb[dfb.scenario == sc]
        for metric in ['mse', 'mae']:
            means, stds = {}, {}
            for m in cols:
                src = gb if (metric == 'mae' and m in gb.method.unique()) else g
                sel = src[src.method == m][metric]
                means[m], stds[m] = sel.mean(), sel.std()
            s = pd.Series(means)
            bolds = top_two(s)
            cells = ' & '.join(fmt(means[m], stds[m], m in bolds) for m in cols)
            pre = f'\\multirow{{2}}{{*}}{{{disp}}}' if metric == 'mse' else ''
            print(f'{pre} & {"MSE" if metric=="mse" else "MAE"} & {cells} \\\\')
        print('\\hline')
    print()


def extra_numbers():
    df1 = pd.read_csv(fr'{exp}\results_new_E1.csv')
    print('===== 叙事数字 =====')
    print('E1 clean BIAM mse:', round(df1[(df1.scenario == "clean") & (df1.method == "BIAM")].mse.mean(), 4))
    print('E1 clean SAM mse:', round(df1[(df1.scenario == "clean") & (df1.method == "SAM")].mse.mean(), 4))
    g = df1[df1.method == 'BIAM'].groupby('scenario')['mae'].mean()
    print('E1 BIAM mae by scenario:', {k: round(v, 3) for k, v in g.items()})
    g2 = df1[df1.method == 'SAM'].groupby('scenario')['mae'].mean()
    print('E1 SAM mae by scenario:', {k: round(v, 3) for k, v in g2.items()})
    g3 = df1[df1.method == 'BIAM-B'].groupby('scenario')['mse'].mean()
    print('E1 BIAM-B mse (ablation):', {k: round(v, 3) for k, v in g3.items()})
    df4 = pd.read_csv(fr'{exp}\results_new_E4.csv')
    for m in ['BIAM', 'Huber', 'BIAM-B']:
        v = df4[(df4.scenario == 'cauchy') & (df4.method == m)].mse.mean()
        print(f'E4 cauchy {m} mse: {v:.3f}')
    # E6/E7 汇总（mean±std）
    for blk, met in [('E6', 'mse'), ('E7', 'f1_macro')]:
        df = pd.read_csv(fr'{exp}\results_biam2_{blk}.csv')
        print(f'----- {blk} ({met}) mean±std -----')
        for (sc, me), g in df.groupby(['scenario', 'method']):
            if me in ('BIAM2', 'BIAM', 'EBM', 'XGBoost', 'CatBoost', 'MLP', 'NAM', 'Lasso', 'SAM'):
                print(f'  {sc:16s} {me:9s} {g[met].mean():.4f} ± {g[met].std():.4f}')
        au = df[df.method == 'BIAM2'].groupby('scenario')['auprc_pair'].agg(['mean', 'std'])
        print('  BIAM2 AUPRC: ' + ', '.join(f'{s}: {r["mean"]:.3f}±{r["std"]:.3f}' for s, r in au.iterrows()))


if __name__ == '__main__':
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if only in ('all', 'e1mse'):
        e1_table('mse', 'tab:synthetic (E1 MSE)')
    if only in ('all', 'e1mae'):
        e1_table('mae', 'tab:mae_syn (E1 MAE)')
    if only in ('all', 'e4'):
        e4_table()
    if only in ('all', 'extra'):
        extra_numbers()
