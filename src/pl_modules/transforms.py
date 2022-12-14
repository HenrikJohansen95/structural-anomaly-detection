import torch

from utils.utils import falling_factorial


class SupportToAnomScore:
    def __init__(self, scaling_coefficient, log_scale=False):
        self.scaling_coefficient = scaling_coefficient
        self.log_scale = log_scale

    def __repr__(self) -> str:
        return "supp_to_anom_score"

    def __call__(self, dataset, training=False):
        if (self.log_scale):
            for sample in dataset.data_list:
                ff = falling_factorial(sample[0].num_nodes, dataset.k)
                sample[2] = - \
                    torch.log(self.scaling_coefficient * sample[2] / ff)
        else:
            for sample in dataset.data_list:
                ff = falling_factorial(sample[0].num_nodes, dataset.k)
                sample[2] = 1 - self.scaling_coefficient * sample[2] / ff


class Standardizer:
    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

    def __repr__(self) -> str:
        return "standardizer"

    def __call__(self, dataset, training=False):
        if training:
            y = torch.cat([e[-1] for e in dataset.data_list])
            self.set_stats(y.mean(), y.std())
        for sample in dataset.data_list:
            sample[-1] = (sample[-1]-self.mean) / self.std


class Normalizer:
    def set_stats(self, minimum, max):
        self.min = minimum
        self.max = max

    def __repr__(self) -> str:
        return "normalizer"

    def __call__(self, dataset, training=False):
        if training:
            y = torch.cat([e[-1] for e in dataset.data_list])
            self.set_stats(y.min(), y.max())

        for sample in dataset.data_list:
            sample[-1] = (sample[-1]-self.min) / \
                (self.max - self.min)


class ThresholdAtDelta:
    def __init__(self, delta) -> None:
        self.delta = delta

    def __repr__(self) -> str:
        return "delta-thresholder"

    def __call__(self, dataset, training=False):
        for sample in dataset.data_list:
            sample[-1] = (sample[-1] >= self.delta).float()


class ThresholdAtFraction:
    def __init__(self, fraction) -> None:
        self.fraction = fraction

    def __repr__(self) -> str:
        return "fraction-thresholder"

    def __call__(self, datasets):
        for dataset in datasets:
            y = torch.cat([e[-1] for e in dataset.data_list])
            delta_idx = int(self.fraction * y.size(0))
            delta = y.sort(descending=True, dim=0)[0][delta_idx]

            for sample in dataset.data_list:
                sample[-1] = (sample[-1] >= delta).float()
