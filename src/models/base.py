"""
Base Few Shot models
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from torch import (
    nn,
    Tensor
)

from src.config import Config, TrainerConfig
from src.schema import Metric
from src.utils import l2, cosine


@dataclass
class FewShotBatchFeather:
    ids: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    label: Optional[Tensor] = None


class FewShotModel(nn.Module):
    def __init__(self, config: Config, train_config: TrainerConfig):
        super().__init__()

        self.config = config
        self.train_config = train_config

    def metrics(self, feature_space: Tensor, support_set_feature_space: Tensor, metric: Metric) -> Tensor:
        """
        run metrics with name
        Args:
            feature_space: the train/input feature space from encoder
            support_set_feature_space: the support set feature space from encoder
            metric: the name of metric distance algo

        Returns: the loss between spaces
        """
        if metric == Metric.L2:
            return l2(feature_space, support_set_feature_space)
        if metric == Metric.Cosine:
            return cosine(feature_space, support_set_feature_space)
        raise NotImplementedError(f'not supported metric<{metric}>')
