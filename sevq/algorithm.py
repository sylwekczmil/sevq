import numpy as np


class SEVQ:

    def __init__(self, *args, **kwargs):
        self.w = None
        self.w_n = None
        self.out_w = None

    def _init_weights(self, x, y):
        self.w_n = np.array([1])
        self.w = np.atleast_2d(x)
        self.out_w = np.array([y])

    def _add_category(self, x, y):
        self.w = np.vstack((self.w, x))
        self.w_n = np.append(self.w_n, 1)
        self.out_w = np.append(self.out_w, y)

    def _match_category(self, x, y=None):
        diff = np.absolute(self.w - x)
        distances = np.linalg.norm(diff, axis=1, ord=2)
        closest_category = np.argmin(distances)
        if y is None:
            return closest_category
        if y != self.out_w[closest_category]:
            return -1
        return closest_category

    def _update_category(self, x, category):
        self.w[category] = self.w[category] + ((x - self.w[category]) / self.w_n[category])
        self.w_n[category] += 1

    def fit(self, x, y, epochs=10, permute=True):
        samples = np.atleast_2d(np.array(x, dtype=float))
        labels = np.atleast_1d(y)

        if self.w is None:
            self._init_weights(samples[0], labels[0])

        _samples = samples
        _labels = labels
        for epoch in range(epochs):
            if permute:
                idx = np.random.permutation(np.arange(len(samples)))
                _samples = samples[idx]
                _labels = labels[idx]

            for _sample, _label in zip(_samples, _labels):
                category = self._match_category(_sample, _label)
                if category == -1:
                    self._add_category(_sample, _label)
                else:
                    self._update_category(_sample, category)

        return self

    def partial_fit(self, x, y):
        return self.fit(x, y, epochs=1, permute=False)

    def predict(self, x):
        samples = np.atleast_2d(x)
        labels = np.full(len(samples), -1)

        if self.w is None:
            return labels

        for i, sample in enumerate(samples):
            category = self._match_category(sample)
            labels[i] = self.out_w[category]
        return labels

    def retrain(self, epochs=10, permute=True):
        _c = SEVQ()
        _c.fit(self.w, self.out_w, epochs, permute)
        self.w = _c.w
        self.w_n = _c.w_n
        self.out_w = _c.out_w
