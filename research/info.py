import itertools

import numpy as np
import pandas as pd

from research.helpers.classifiers import CLASSIFIERS, CLASSIFIERS_FOR_NORMALIZED
from research.helpers.data import datasets, RESEARCH_DIR
from research.helpers.util import to_latex, seed_everything


def dataset_info():
    print(f'Processing datasets info')
    df = pd.DataFrame(columns=['Dataset', 'Records', 'Attributes', 'Classes', 'Imbalance Score'])

    for dataset_idx, dataset in enumerate(datasets(only_numerical=True)):
        x, y = dataset.data()

        unique, counts = np.unique(y, return_counts=True)
        imbalances = []
        for a, b in itertools.combinations(counts, 2):
            imbalances.append(min(a, b) / max(a, b))
        imbalance = f'{sum(imbalances) / len(imbalances):.4f}'
        row = {
            'Dataset': dataset.name,
            'Records': x.shape[0],
            'Attributes': x.shape[1],
            'Classes': len(set(y)),
            'Imbalance Score': imbalance
        }
        df = df.append(row, ignore_index=True)

    df.index += 1
    RESEARCH_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(RESEARCH_DIR.joinpath(f'datasets.csv'), index=True)
    f = RESEARCH_DIR.joinpath(f'datasets.tex').open('w')
    tex = to_latex(df, escape=False,
                   caption='Datasets used to perform experiments',
                   label='tab:datasets')
    f.write(tex)


def algorithm_info():
    print(f'Processing algorithms info')
    df = pd.DataFrame(columns=['Acronym', 'Name', 'Library', 'Type'])
    for cn, c in CLASSIFIERS + CLASSIFIERS_FOR_NORMALIZED:
        cz = c(2, 2).__class__
        library = cz.__module__.split('.')[0]
        name = cz.__name__
        if cn == 'SEVQ' or cn == 'SFAM' or cn == 'EVQ':
            library = 'custom'
        if cn == 'XGB':
            library = 'xgboost'
        name = name.replace('Classifier', '')
        if cn != 'SEVQ':
            row = {
                'Acronym': cn,
                'Name': name,
                'Library': library,
                'Type': 'classic' if (library == 'sklearn' or library == 'xgboost') else 'incremental'
            }
            df = df.append(row, ignore_index=True)

    df = df.drop_duplicates()
    df.sort_values(['Type', 'Acronym'], inplace=True)
    df = df.reset_index(drop=True)
    df.index += 1
    df.to_csv(RESEARCH_DIR.joinpath(f'algorithms.csv'), index=True)
    f = RESEARCH_DIR.joinpath(f'algorithms.tex').open('w')
    tex = to_latex(df, escape=False,
                   caption='Algorithms used to perform experiments',
                   label='tab:algorithm')
    f.write(tex)


if __name__ == '__main__':
    seed_everything()
    dataset_info()
    algorithm_info()
