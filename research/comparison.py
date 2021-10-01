import inspect
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from joblib import delayed, Parallel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from research.helpers.classifiers import SimplifiedFuzzyARTMAP, CLASSIFIERS, CLASSIFIERS_FOR_NORMALIZED
from research.helpers.data import datasets, RESEARCH_DIR
from research.helpers.util import auc, seed_everything, to_latex
from sevq.algorithm import SEVQ


def _process_comparison_single(c, c_n, dataset, fold):
    cls = c(fold.x_train.shape[1], len(fold.labels))
    algorithm = c_n
    train_start_time = time.time()
    labels = fold.labels
    if 'VQ' in algorithm:
        cls.fit(fold.x_train, fold.y_train, epochs=10)
    elif 'classes' in inspect.getfullargspec(cls.fit).args:
        cls.fit(fold.x_train, fold.y_train, classes=labels.tolist())
    else:
        cls.fit(fold.x_train, fold.y_train)
    train_time = (time.time() - train_start_time)
    pred_start_time = time.time()
    pred = cls.predict(fold.x_test)
    pred_time = (time.time() - pred_start_time)
    number_of_categories = 0
    if cls.__class__ == SEVQ or cls.__class__ == SimplifiedFuzzyARTMAP:
        number_of_categories = len(cls.out_w)
    return {
        'dataset': dataset.name,
        'algorithm': algorithm,
        'number_of_classes': len(set(fold.y_train)),
        'number_of_categories': number_of_categories,
        'train_size': len(fold.x_train),
        'test_size': len(fold.x_test),
        'cv_idx': fold.index,
        'accuracy': accuracy_score(fold.y_test, pred),
        'precision': precision_score(fold.y_test, pred, average='weighted', labels=labels, zero_division=0),
        'recall': recall_score(fold.y_test, pred, average='weighted', labels=labels, zero_division=0),
        'f1': f1_score(fold.y_test, pred, average='weighted', labels=labels, zero_division=0),
        'auc': auc(fold.y_test, pred, average='weighted', multi_class='ovo', labels=labels),
        'train_time': train_time,
        'pred_time': pred_time
    }


def process_comparison(normalized: bool):
    name = 'comparison'
    classifiers = CLASSIFIERS
    if normalized:
        name = 'comparison_normalized'
        classifiers = CLASSIFIERS_FOR_NORMALIZED
    print(f'Processing {name}')
    RESEARCH_DIR.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(
        columns=['dataset', 'algorithm', 'number_of_classes', 'number_of_categories',
                 'train_size', 'test_size', 'cv_idx',
                 'accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time', 'pred_time'])

    count = 0
    for dataset_idx, dataset in enumerate(datasets(only_numerical=True)):
        for fold in dataset.data_10_fold():
            if normalized:
                fold.normalize()
            print(f'Processing dataset {dataset_idx} {dataset.name}, fold {fold.index}')
            rows = Parallel(n_jobs=len(classifiers))(
                delayed(_process_comparison_single)(c, c_n, dataset, fold) for c_n, c in classifiers)
            df = df.append(rows, ignore_index=True)

        df = df.sort_values(by=['dataset', 'algorithm', 'cv_idx'])
        count += 1
        df.to_csv(RESEARCH_DIR.joinpath(f'{name}_{count}.csv'), index=False)
        if count > 1:
            prev_file = RESEARCH_DIR.joinpath(f'{name}_{count - 1}.csv')
            if os.path.isfile(prev_file):
                os.remove(prev_file)
    df.to_csv(RESEARCH_DIR.joinpath(f'{name}.csv'), index=False)


def process_comparison_results(normalized: bool):
    name = 'comparison_normalized' if normalized else 'comparison'
    print(f'Processing {name} results')
    df = pd.read_csv(RESEARCH_DIR.joinpath(f'{name}.csv'))

    gb = ['algorithm']
    dfg = df.groupby(gb)
    df = dfg.mean()
    dfg_std = dfg.std()

    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    df_csv = df.copy(deep=True)
    for metric in metrics:
        df_csv.insert(list(df_csv.columns).index(metric) + 1, f'{metric} +/-', dfg_std[metric])

    columns = list(df_csv.columns[13:15]) + list(df_csv.columns[5:13])
    df_csv = df_csv[columns]
    df_csv = df_csv.sort_values(by=['auc', 'accuracy'], ascending=False)

    df_csv.reset_index(inplace=True)
    df_csv.index += 1
    df_csv.to_csv(RESEARCH_DIR.joinpath(f'{name}_result.csv'))

    df_tex = df_csv.copy(deep=True)
    for metric in metrics:
        metric_pm = metric + ' +/-'
        df_tex[metric] = df_tex[metric].apply(lambda x: "{:1.3f}".format(x))
        df_tex[metric_pm] = df_tex[metric_pm].apply(lambda x: "{:1.3f}".format(x))
        df_tex[metric] = df_tex[metric].astype(str) + '$\pm$' + df_tex[metric_pm].astype(str)
        del df_tex[metric_pm]

    f = RESEARCH_DIR.joinpath(f'{name}_result.tex').open('w')
    tex = to_latex(df_tex, escape=False,
                   caption=f'Results of comparison of {"incremental" if normalized else "traditional"} algorithms',
                   label=f'tab:{"Incremental" if normalized else "Traditional"}_comparison',
                   )
    f.write(tex)


def process_comparison_results_charts(normalized: bool):
    name = 'comparison_normalized' if normalized else 'comparison'
    print(f'Processing {name} results chart')
    df_results = pd.read_csv(RESEARCH_DIR.joinpath(f'{name}.csv'))
    comparison_plot_dir = RESEARCH_DIR.joinpath('comparison_plot')
    comparison_plot_dir.mkdir(exist_ok=True, parents=True)

    def boxplot_sorted(df, by, column, file_suffix=''):
        df2 = pd.DataFrame({col: vals[column] for col, vals in df.groupby(by)})
        meds = df2.median().sort_values(ascending=False)
        df2[meds.index].boxplot(return_type="axes")
        plt.title('')
        plt.suptitle('')
        ylabel = 'AUC' if metric == 'auc' else metric.capitalize()
        plt.ylabel(ylabel)
        plt.xlabel('Algorithm')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(comparison_plot_dir.joinpath(f'{name}_{metric}{file_suffix}.eps'))
        plt.savefig(comparison_plot_dir.joinpath(f'{name}_{metric}{file_suffix}.png'))
        plt.close()

    for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        boxplot_sorted(df_results, column=metric, by='algorithm', file_suffix='_per_fold')
        boxplot_sorted(df_results.groupby(['algorithm', 'dataset']).mean().reset_index(level=0),
                       column=metric, by='algorithm', file_suffix='')


def process_comparison_results_median(normalized: bool):
    name = 'comparison_normalized' if normalized else 'comparison'
    df = pd.read_csv(RESEARCH_DIR.joinpath(f'{name}.csv'))

    gb = ['algorithm']
    dfg = df.groupby(gb)
    df = dfg.median()

    df_csv = df.copy(deep=True)
    df_csv = df_csv[df_csv.columns[5:10]].sort_values(by=['accuracy'], ascending=False)
    df_csv.to_csv(RESEARCH_DIR.joinpath(f'{name}_result_median.csv'))

    f = RESEARCH_DIR.joinpath(f'{name}_result_median.tex').open('w')
    tex = to_latex(df, escape=False,
                   caption=f'Results of median comparison of {"incremental" if normalized else "traditional"} algorithms',
                   label=f'tab:{"Incremental" if normalized else "Traditional"}_comparison_median',
                   )
    f.write(tex)


def process_comparison_result_winners_for_metric(normalized: bool, metric: str):
    name = 'comparison_normalized' if normalized else 'comparison'
    df = pd.read_csv(RESEARCH_DIR.joinpath(f'{name}.csv'))
    algorithms = df['algorithm'].unique()
    places = [i for i in range(3)]

    winner_dir = RESEARCH_DIR.joinpath('winner').joinpath(metric)
    winner_dir.mkdir(exist_ok=True, parents=True)

    def count_places(place=0):
        count = {a: 0 for a in algorithms}
        names = {a: [] for a in algorithms}
        for dataset, df_d in df.groupby(['dataset']):
            df_d_a_m = df_d.groupby(['algorithm']).mean().sort_values(by=[metric], ascending=False)
            best = df_d_a_m.iloc[place]
            count[best.name] += 1
            names[best.name].append(dataset)

        return count, names

    counts = []
    sevq_names = []
    for c, n in [count_places(i) for i in places]:
        counts.append(c)
        sevq_names.append(n['SEVQ'])

    rows = []
    for algorithm in algorithms:
        row = [algorithm]
        for p in places:
            row.append(counts[p][algorithm])
        rows.append(row)

    df_r = pd.DataFrame(columns=['algorithm', '1st', '2nd', '3rd'], data=rows)
    df_r = df_r.sort_values(by=['1st'], ascending=False)
    df_r.reset_index(drop=True, inplace=True)
    df_r.index += 1
    df_r.to_csv(winner_dir.joinpath(f'{name}_result.csv'), index=True)
    winner_dir.joinpath(f'{name}_result.tex').open('w').write(
        to_latex(df_r, index=True,
                 caption=f'Ranking of compared {"incremental" if normalized else "traditional"} algorithms for {metric}',
                 label=f'tab:{"Incremental" if normalized else "Traditional"}_places_{metric}',
                 ))

    names_result = ''
    for i, nl in enumerate(sevq_names):
        names_result += ','.join(nl)
        names_result += '\n'

    # best SEVQ
    winner_dir.joinpath(f'{name}_result_top_3_names.txt').open('w').write(names_result)

    # worst SEVQ
    _, nl = count_places(len(algorithms) - 1)
    winner_dir.joinpath(f'{name}_result_worst_names.txt').open('w').write(','.join(nl['SEVQ']))

    return df_r


def process_comparison_result_winners(normalized: bool):
    name = 'comparison_normalized' if normalized else 'comparison'
    print(f'Processing {name} results winners')
    auc_wins = process_comparison_result_winners_for_metric(normalized, 'auc').sort_values(by=['algorithm'])
    acc_wins = process_comparison_result_winners_for_metric(normalized, 'accuracy').sort_values(by=['algorithm'])
    wins_df = auc_wins[['algorithm']].copy()

    for c in acc_wins.columns[1:]:
        wins_df['auc ' + c] = auc_wins[c].values

    for c in acc_wins.columns[1:]:
        wins_df['accuracy ' + c] = acc_wins[c].values

    winner_dir = RESEARCH_DIR.joinpath('winner')
    winner_dir.mkdir(exist_ok=True, parents=True)
    wins_df = wins_df.sort_values(by=wins_df.columns[1:].values.tolist(), ascending=False)
    wins_df.reset_index(drop=True, inplace=True)
    wins_df.index += 1
    wins_df.to_csv(winner_dir.joinpath(f'{name}.csv'), index=True)
    winner_dir.joinpath(f'{name}.tex').open('w').write(
        to_latex(wins_df, index=True,
                 caption=f'Ranking of compared {"incremental" if normalized else "traditional"} algorithms',
                 label=f'tab:{"Incremental" if normalized else "Traditional"}_places',
                 ))


if __name__ == '__main__':
    seed_everything()
    for normalized in [False, True]:
        # process_comparison(normalized=normalized)
        # process_comparison_results(normalized=normalized)
        # process_comparison_results_charts(normalized=normalized)
        # process_comparison_results_median(normalized=normalized)
        process_comparison_result_winners(normalized=normalized)
