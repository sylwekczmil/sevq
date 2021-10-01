from collections import defaultdict
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from evq.algorithm import EVQ
from matplotlib.image import imread
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.metrics import accuracy_score, silhouette_score

from research.helpers.data import RESEARCH_DIR
from research.helpers.generators import make_spirals
from research.helpers.plot import plot, plot_incremental, plot_evq
from research.helpers.util import seed_everything, to_latex
from sevq.algorithm import SEVQ

seed_everything()


def process_generated():
    print(f'Processing generated charts')
    generated_dir = RESEARCH_DIR.joinpath('generated')
    generated_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(
        columns=['dataset', 'records', '0.9 accuracy after', 'max accuracy', 'max accuracy after',
                 'pearsonr', 'silhouette'])

    for idx, ((x, y), name) in enumerate([
        (make_circles(noise=0.05, random_state=1), 'circles'),
        (make_moons(noise=0.05, random_state=1), 'moons'),
        (make_blobs(centers=3, random_state=1), '3_blobs'),
        (make_spirals(), 'spirals'),
        # (make_checkerboard(), 'checkerboard'),
        # (make_checkerboard(padding=0.2), 'checkerboard_padding')
    ]):

        net = SEVQ()
        indexes = []
        class_indexes = defaultdict(list)
        class_accuracy = defaultdict(list)
        accuracies = []
        first_90_idx = None
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)
        for s_idx, (s_x, s_y) in enumerate(zip(x, y)):
            net.partial_fit(s_x, s_y)
            yp = net.predict(x)
            acc = accuracy_score(y, yp)
            indexes.append(s_idx)
            accuracies.append(acc)
            class_indexes[s_y].append(len(class_indexes[s_y]))
            class_accuracy[s_y].append(acc)
            if first_90_idx is None and acc >= 0.9:
                first_90_idx = s_idx

        r = pearsonr(x[:, 0], x[:, 1])[0]
        s = silhouette_score(x, y)
        best_idx = np.argmax(accuracies)
        row = {
            'dataset': name,
            'records': len(y),
            '0.9 accuracy after': first_90_idx + 1,
            'max accuracy': np.max(accuracies),
            'max accuracy after': best_idx + 1,
            'pearsonr': f'{r:.4f}',
            'silhouette': f'{s:.4f}',
        }
        df = df.append(row, ignore_index=True)
        title = f'{name} categories/samples {len(net.out_w)}/{len(x)}'
        plot(net, x, y, path=generated_dir.joinpath(f'{name}.eps'), title=title)
        plot(net, x, y, path=generated_dir.joinpath(f'{name}.png'), title=title)
        plot_incremental(name, title, indexes, accuracies, generated_dir)
        for k in class_indexes.keys():
            plot_incremental(f'{name}-class-{k}', f'{title} class-{k}', class_indexes[k], class_accuracy[k],
                             generated_dir)
    df.index += 1
    df.to_csv(generated_dir.joinpath(f'generated_datasets.csv'), index=True)
    f = generated_dir.joinpath(f'generated_datasets.tex').open('w')
    tex = to_latex(df, escape=False,
                   caption=f'Generated algorithms results',
                   label=f'tab:generated',
                   )
    f.write(tex)


def process_generated_evq():
    print(f'Processing generated evq charts')
    generated_dir = RESEARCH_DIR.joinpath('generated_evq')
    generated_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(
        columns=['dataset', 'records', '0.9 accuracy after', 'max accuracy', 'max accuracy after',
                 'pearsonr', 'silhouette'])

    for idx, ((x, y), name) in enumerate([
        # (make_circles(noise=0.05, random_state=1), 'circles'),
        # (make_moons(noise=0.05, random_state=1), 'moons'),
        (make_blobs(centers=3, random_state=1), '3_blobs'),
        # (make_spirals(), 'spirals'),
        # (make_checkerboard(), 'checkerboard'),
        # (make_checkerboard(padding=0.2), 'checkerboard_padding')
    ]):

        net = EVQ(number_of_classes=len(np.unique(y)), vigilance=0.2)
        indexes = []
        class_indexes = defaultdict(list)
        class_accuracy = defaultdict(list)
        accuracies = []
        first_90_idx = None
        min_max_scaler = preprocessing.MinMaxScaler()
        x = min_max_scaler.fit_transform(x)
        for s_idx, (s_x, s_y) in enumerate(zip(x, y)):
            net.partial_fit(s_x, s_y)
            yp = net.predict(x)
            acc = accuracy_score(y, yp)
            indexes.append(s_idx)
            accuracies.append(acc)
            class_indexes[s_y].append(len(class_indexes[s_y]))
            class_accuracy[s_y].append(acc)
            if first_90_idx is None and acc >= 0.9:
                first_90_idx = s_idx

        r = pearsonr(x[:, 0], x[:, 1])[0]
        s = silhouette_score(x, y)
        best_idx = np.argmax(accuracies)
        row = {
            'dataset': name,
            'records': len(y),
            # '0.9 accuracy after': first_90_idx + 1,
            'max accuracy': np.max(accuracies),
            'max accuracy after': best_idx + 1,
            'pearsonr': f'{r:.4f}',
            'silhouette': f'{s:.4f}',
        }
        df = df.append(row, ignore_index=True)
        title = f'{name} clusters/samples {len(net.clusters)}/{len(x)}'
        plot_evq(net, x, y, path=generated_dir.joinpath(f'{name}.eps'), title=title)
        plot_evq(net, x, y, path=generated_dir.joinpath(f'{name}.png'), title=title)
        plot_incremental(name, title, indexes, accuracies, generated_dir)
        for k in class_indexes.keys():
            plot_incremental(f'{name}-class-{k}', f'{title} class-{k}', class_indexes[k], class_accuracy[k],
                             generated_dir)
    df.index += 1
    df.to_csv(generated_dir.joinpath(f'generated_datasets.csv'), index=True)
    f = generated_dir.joinpath(f'generated_datasets.tex').open('w')
    tex = to_latex(df, escape=False,
                   caption=f'Generated algorithms results',
                   label=f'tab:generated',
                   )
    f.write(tex)


def process_generated_comparison():
    f, axarr = plt.subplots(8, 2, figsize=(20, 80))
    gd1 = RESEARCH_DIR.joinpath('generated')
    gd2 = RESEARCH_DIR.joinpath('generated_evq')
    for i_name, name in enumerate(['circles', 'moons', '3_blobs', 'spirals']):
        img1 = imread(gd1.joinpath(name + '.png'))
        inc_img1 = imread(gd1.joinpath(name + '-incremental.png'))
        img2 = imread(gd2.joinpath(name + '.png'))
        inc_img2 = imread(gd2.joinpath(name + '-incremental.png'))
        i = i_name * 2
        axarr[i, 0].imshow(img1)
        axarr[i, 0].axis('off')
        if i_name == 0:
            axarr[i, 0].title.set_text('SEVQ')
            axarr[i, 0].title.set_fontsize(30)
        axarr[i + 1, 0].imshow(inc_img1)
        axarr[i + 1, 0].axis('off')
        axarr[i, 1].imshow(img2)
        axarr[i, 1].axis('off')
        if i_name == 0:
            axarr[i, 1].title.set_text('EVQ')
            axarr[i, 1].title.set_fontsize(30)
        axarr[i + 1, 1].imshow(inc_img2)
        axarr[i + 1, 1].axis('off')

    plt.tight_layout()
    name = 'generated_comparison'
    plt.savefig(RESEARCH_DIR.joinpath(f'{name}.eps'))
    plt.savefig(RESEARCH_DIR.joinpath(f'{name}.png'))
    plt.close()


if __name__ == '__main__':
    seed_everything()
    process_generated()
    process_generated_evq()
    process_generated_comparison()
