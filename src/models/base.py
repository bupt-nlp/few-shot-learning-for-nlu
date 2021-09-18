"""
Base Few Shot models
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from torch import (
    nn,
    Tensor
)

from src.schema import Config, TrainConfig, Metric


@dataclass
class FewShotBatchFeather:
    ids: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
    label: Optional[Tensor] = None


class FewShotModel(nn.Module):
    def __init__(self, config: Config, train_config: TrainConfig):
        super().__init__()

        self.config = config
        self.train_config = train_config

    def forward(self, train_feature: FewShotBatchFeather, support_set_feature: FewShotBatchFeather):
        raise NotImplementedError

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
            return self.l2_metric(feature_space, support_set_feature_space)

    @staticmethod
    def l2_metric(feature_space: Tensor, support_set_feature_space: Tensor) -> Tensor:
        """
        run L2 metric between train/input feature space and support set feature space
        Args:
            feature_space:
            support_set_feature_space:

        Returns:

        """
        
