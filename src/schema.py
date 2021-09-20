from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict

from dataclasses_json import dataclass_json
from tap import Tap
from transformers import PreTrainedTokenizer


class Metric(Enum):
    L2 = 'l2'
    Cosine = 'Cosine'


class FewShotDataSet(Enum):
    Snips = 'snips'
    Clinc = 'clinc'


class Config(Tap):
    """
    Config object which can read configuration from command parameter & json file
    """
    dataset: FewShotDataSet = 'clinc'  # snips
    metric: Metric = Metric.Cosine  # metric
    n_way_train: int = 5  # number of support examples per class for training tasks
    n_way_validation: int = 5  # number of support examples per class for validation task

    k_shot_train: int = 5  # number of classes for training tasks
    k_shot_validation: int = 5  # number of classes for validation tasks

    pretrain_model: str = ''

    train_file: str = './data/'

    @staticmethod
    def from_file(file: str) -> Config:
        # 1. read the configuration from file
        with open(file, 'r', encoding='utf-8') as f:
            configuration = json.load(f)

        args = ['']
        for key, value in configuration.items():
            args.append(f'--{key}')
            args.append(value)

        config = Config().parse_args(args, known_only=True)
        return config


class TrainConfig(Tap):
    epoch: int
    batch_size: int
    learning_rate: float
    num_task: int


@dataclass
class FewShotFeature:
    id: int
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]

    label: int


@dataclass
class FewShotExample:
    id: int
    label: Union[str, int]
    raw_text: str
    domain: str

    def to_feature(self, tokenizer: PreTrainedTokenizer, label2id: Dict[str, int]) -> FewShotFeature:
        """convert few-shot example to feature"""
        fields = tokenizer(
            self.raw_text, add_special_tokens=True,
            max_length=20, return_attention_mask=True,
            return_token_type_ids=True,
            return_length=True, return_special_tokens_mask=True
        )
        return FewShotFeature(
            id=self.id,
            input_ids=fields.input_ids,
            attention_mask=fields.attention_mask,
            token_type_ids=fields.token_type_ids,
            label=label2id[self.label]
        )


@dataclass_json
@dataclass
class TextClassificationInputExample:
    raw_text: str
    label: str

    example_id: int = 0


@dataclass_json
@dataclass
class SequenceTaggingInputExample:
    raw_text: str
    labels: List[str]

    example_id: int = 0
