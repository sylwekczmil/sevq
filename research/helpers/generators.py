import numpy as np


def make_spirals(n_samples=100, shuffle=True):
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n_samples, 2)
    y_a = np.zeros(n_samples)

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n_samples, 2)
    y_b = np.ones(n_samples)

    x = np.concatenate((x_a, x_b))
    y = np.concatenate((y_a, y_b))
    if shuffle:
        p = np.random.permutation(len(y))
        x = x[p]
        y = y[p]
    return x, y.astype(int)


def make_checkerboard(n_samples=450, padding=0.01, shuffle=True):
    n_samples_per_square = n_samples // 9

    def g(_x_diff, _y_diff):
        a = np.random.rand(n_samples_per_square, 2)
        a[a < padding] = padding
        a[a > 1 - padding] = 1 - padding
        a[:, 0] += _x_diff
        a[:, 1] += _y_diff
        return a

    x_list = []
    y_list = []
    for x_diff in range(3):
        for y_diff in range(3):
            y_list.append(np.ones(n_samples_per_square) * ((x_diff + y_diff) % 2))
            x_list.append(g(x_diff, y_diff))

    x = np.concatenate(x_list)
    y = np.concatenate(y_list)
    if shuffle:
        p = np.random.permutation(len(y))
        x = x[p]
        y = y[p]
    return x, y
