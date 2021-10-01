import pandas as pd

from research.helpers.data import RESEARCH_DIR


def process_times(normalized: bool):
    name = 'comparison_normalized' if normalized else 'comparison'
    print(f'Processing {name} results')
    df = pd.read_csv(RESEARCH_DIR.joinpath(f'{name}.csv'))
    gb = ['algorithm']
    dfg = df.groupby(gb)
    df = dfg.mean()
    columns = ['train_time', 'pred_time']
    df_csv = df[columns]
    df_csv = df_csv.sort_values(by=['train_time'], ascending=True)
    df_csv.to_csv(RESEARCH_DIR.joinpath(f'{name}_time_train.csv'))
    df_csv = df_csv.sort_values(by=['pred_time'], ascending=True)
    df_csv.to_csv(RESEARCH_DIR.joinpath(f'{name}_time_pred.csv'))
    print('')


if __name__ == '__main__':
    for normalized in [False, True]:
        process_times(normalized)
