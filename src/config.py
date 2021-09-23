from __future__ import annotations
import os
from typing import Optional, List, Union
import json
from tap import Tap

from src.schema import FewShotDataSet, Metric


SEED = 200
root_dir = os.path.dirname(os.path.dirname(__file__))


class BaseConfig(Tap):

    @classmethod
    def _load_args(cls, file: str, field: Optional[str] = None) -> List[str]:
        """
        load args from json file which only include key, value configuration

        Args:
            file: the json file which contains the
            field: sub field in configuration file

        Returns: the final args which the tap obj will read from

        """
        # 1. read the configuration from file
        with open(file, 'r', encoding='utf-8') as f:
            configuration = json.load(f)
            if field and field in configuration:
                configuration = configuration[field]

        args = []
        for key, value in configuration.items():
            if isinstance(value, dict):
                continue
            args.append(f'--{key}')
            args.append(str(value))
        return args


class Config(BaseConfig):
    """
    Config object which can read configuration from command parameter & json file
    """
    dataset: Union[str, FewShotDataSet] = 'clinc'  # snips
    metric: Union[str, Metric] = 'cosine'  # metric
    n_way: int = 5  # number of support examples per class for training tasks
    k_shot: int = 5  # number of support examples per class for validation task

    num_labels: int = 3

    pretrain_model: str = 'bert-base-uncased'

    train_file: str = './data/'
    evaluation_file: str = './data/'
    test_file: str = './data/'

    @classmethod
    def from_file(cls, file: str) -> Config:
        args = cls._load_args(file, 'base')
        return Config().parse_args(args, known_only=True)


class TrainerConfig(BaseConfig):
    epoch: int
    batch_size: int
    learning_rate: float
    warmup_rate: float

    @classmethod
    def from_file(cls, file: str) -> TrainerConfig:
        args = cls._load_args(file, 'trainer')
        return TrainerConfig().parse_args(args, known_only=True)
