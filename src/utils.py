from torch import Tensor
import torch

from src.schema import Metric


def get_metric(query_set: Tensor, support_set: Tensor, metric: Metric) -> Tensor:
    """
    run metrics with name
    Args:
        query_set: the train/input feature space from encoder
        support_set: the support set feature space from encoder

        metric: the name of metric distance algo

    Returns: the loss between spaces
    """
    if metric == Metric.L2:
        return l2(query_set, support_set)
    if metric == Metric.Cosine:
        return cosine(query_set, support_set)
    raise NotImplementedError(f'not supported metric<{metric}>')


def l2(x: Tensor, y: Tensor) -> Tensor:
    return torch.pow(x - y, 2)


def cosine(x: Tensor, y: Tensor) -> Tensor:
    return torch.cosine_similarity(x, y, dim=-1)