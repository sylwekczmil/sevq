import itertools

import numpy as np
from evq.algorithm import EVQ
from matplotlib import pyplot as plt

class_marks = ['1', '2', '3', '4']
cat_marks = ['o', 's', 'D', 'p']


def plot(net, x, y, feature_names=None, show=False, max_number_of_features_to_plot=4, path=None, title=''):
    colors = list('rgbcmyk' * 2)
    number_of_features = len(net.w[0])
    if number_of_features < max_number_of_features_to_plot:
        max_number_of_features_to_plot = number_of_features
    for c1, c2 in list(itertools.combinations(range(max_number_of_features_to_plot), 2)):
        fig, ax = plt.subplots(1)
        if feature_names:
            ax.set_xlabel(feature_names[c1])
            ax.set_ylabel(feature_names[c2])
        xt = x.T
        wt = net.w.T

        for c in set(y):
            c = int(c)
            idxs = np.where(y == c)
            d0 = xt[c1][idxs]
            d1 = xt[c2][idxs]
            ax.plot(d0, d1, colors[c] + class_marks[c], label=f'class {str(c)} sample')
            idxs = np.where(net.out_w == c)
            d0 = wt[c1][idxs]
            d1 = wt[c2][idxs]
            ax.plot(d0, d1, colors[c] + cat_marks[c], label=f'class {str(c)} category')
        ax.legend()
        plt.title(title)
        if path:
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()


def plot_evq(net: EVQ, x, y, feature_names=None, show=False, max_number_of_features_to_plot=4, path=None, title=''):
    colors = list('rgbcmyk' * 2)
    number_of_features = len(net.clusters[0])
    if number_of_features < max_number_of_features_to_plot:
        max_number_of_features_to_plot = number_of_features
    for c1, c2 in list(itertools.combinations(range(max_number_of_features_to_plot), 2)):
        fig, ax = plt.subplots(1)
        if feature_names:
            ax.set_xlabel(feature_names[c1])
            ax.set_ylabel(feature_names[c2])
        xt = x.T
        wt = net.clusters.T


        cluster_classes = np.apply_along_axis(func1d=net.predict, axis=1, arr=net.clusters).flatten()
        for c in set(y):
            c = int(c)
            idxs = np.where(y == c)
            d0 = xt[c1][idxs]
            d1 = xt[c2][idxs]
            ax.plot(d0, d1, colors[c] + class_marks[c], label=f'class {str(c)} sample')
            idxs = np.where(cluster_classes == c)
            d0 = wt[c1][idxs]
            d1 = wt[c2][idxs]
            ax.plot(d0, d1, colors[c] + cat_marks[c], label=f'cluster predicted as class {str(c)}')
        ax.legend()
        plt.title(title)
        if path:
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()


def plot_incremental(name, title, indexes, accuracies, generated_dir):
    plt.plot(indexes, accuracies)
    plt.title(title)
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
    ax = plt.gca()
    ax.set_ylim([-0.05, 1.05])
    plt.savefig(generated_dir.joinpath(f'{name}-incremental.eps'))
    plt.savefig(generated_dir.joinpath(f'{name}-incremental.png'))
    plt.close()

def plot3d(net, x=None, y=None, feature_names=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = list('rgb')

    xt = x.T
    wt = net.w.T
    for c in set(y):
        c = int(c)
        idxs = np.where(y == c)
        d0 = xt[0][idxs]
        d1 = xt[1][idxs]
        d2 = xt[2][idxs]
        ax.scatter(d0, d1, d2, label=f'class {str(c)} record', marker='x', c=colors[c])

        idxs = np.where(net.out_w == c)
        d0 = wt[0][idxs]
        d1 = wt[1][idxs]
        d2 = wt[2][idxs]
        ax.scatter(d0, d1, d2, label=f'class {str(c)} category', marker='o', c=colors[c])

    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
    ax.legend()
    if show:
        plt.show()
    plt.close()