import pandas as pd
from scipy.stats import wilcoxon

from research.helpers.data import RESEARCH_DIR
from research.helpers.util import to_latex


def bold_large_p_value(data, format_string="%.4f"):
    if data > 0.05:
        return "\\textbf{%s}" % format_string % data

    return "%s" % format_string % data


def process_wilcoxon(normalized=True):
    name = 'comparison_normalized' if normalized else 'comparison'
    print(f'Processing {name} wilcoxon')
    acc_df = process_wilcoxon_for_metric(normalized, 'accuracy').sort_values(by=['Algorithm'])
    auc_df = process_wilcoxon_for_metric(normalized, 'auc').sort_values(by=['Algorithm'])

    r_df = acc_df[['Algorithm']].copy()

    for c in acc_df.columns[2:]:
        r_df['accuracy ' + c] = acc_df[c].values

    for c in auc_df.columns[2:]:
        r_df['auc ' + c] = auc_df[c].values

    wilcoxon_dir = RESEARCH_DIR.joinpath('wilcoxon')
    wilcoxon_dir.mkdir(exist_ok=True, parents=True)
    r_df.reset_index(drop=True, inplace=True)
    r_df.index += 1
    r_df.to_csv(wilcoxon_dir.joinpath(f'{name}.csv'), index=True)

    for col in ['accuracy p-value', 'auc p-value']:
        r_df[col] = r_df[col].apply(lambda data: bold_large_p_value(data))

    wilcoxon_dir.joinpath(f'{name}.tex').open('w').write(
        to_latex(r_df, index=True, escape=False,
                 caption=f'Comparison of {"incremental" if normalized else "traditional"} classifiers and SEVQ with Wilcoxon’s signed-rank test',
                 label=f'tab:{"Incremental" if normalized else "Traditional"}_wilcoxon_comparison',
                 ))


def process_wilcoxon_for_metric(normalized: bool, metric: str):
    wilcoxon_dir = RESEARCH_DIR.joinpath('wilcoxon')
    wilcoxon_dir.mkdir(exist_ok=True, parents=True)
    name = 'comparison_normalized' if normalized else 'comparison'
    metric_dir = wilcoxon_dir.joinpath(metric)
    metric_dir.mkdir(exist_ok=True, parents=True)
    df_r = pd.DataFrame(columns=['SEVQ', 'Algorithm',
                                 # 'w',
                                 'p-value'])
    df = pd.read_csv(RESEARCH_DIR.joinpath(f'{name}.csv'))
    algorithms = list(df.algorithm.unique())
    algorithms.remove('SEVQ')
    mm_df = df[df.algorithm == 'SEVQ']
    for algorithm in algorithms:
        a_df = df[df.algorithm == algorithm]
        w, p = wilcoxon(mm_df[metric], a_df[metric])
        row = {
            'SEVQ': 'SEVQ',
            'Algorithm': algorithm,
            # 'w': w,
            'p-value': p,

        }
        df_r = df_r.append(row, ignore_index=True)

    df_r.reset_index(drop=True, inplace=True)
    df_r.index += 1
    df_r.to_csv(metric_dir.joinpath(f'{name}_result.csv'), index=True)
    f = metric_dir.joinpath(f'{name}_result.tex').open('w')
    f.write(to_latex(df_r, index=True, escape=False,
                     caption=f'Comparison of {"incremental" if normalized else "traditional"} classifiers and SEVQ with Wilcoxon’s signed-rank test for {metric}',
                     label=f'tab:{"Incremental" if normalized else "Traditional"}_wilcoxon_{metric}_comparison',
                     )
            )
    return df_r


if __name__ == '__main__':
    seed_everything()
    for normalized in [False, True]:
        process_wilcoxon(normalized)
