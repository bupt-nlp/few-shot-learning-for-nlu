from __future__ import annotations

import os.path

from pytest import fixture
from pytorch_lightning import Trainer
from src.config import Config, TrainerConfig
from src.config import root_dir


@fixture
def file() -> str:
    return os.path.join(root_dir, 'tests/data/config.json')


def test_config(file: str):
    config = Config.from_file(file)

    assert config.dataset == 'clinc'
    assert config.metric == 'cosine'


def test_train_config(file: str):
    train_config = TrainerConfig.from_file(file)
    assert train_config.epoch == 10


def test_trainer_with_config(file: str):
    train_config = TrainerConfig.from_file(file)
    trainer = Trainer.from_argparse_args(train_config)
    assert trainer.max_epochs


