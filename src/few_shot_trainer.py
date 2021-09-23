"""

"""
from __future__ import annotations
from pytorch_lightning import Trainer

from src.config import Config, TrainerConfig
from src.dataset_reader.text_classification import (
    CLINCDataSetReader
)


def main(config_file: str):
    # 1. init the basic configuration
    config, train_config = Config.from_file(config_file), TrainerConfig.from_file(config_file)

    # 2. get dataset loader
    dataset_loader = CLINCDataSetReader(config)

    # 3. train the model
    trainer = Trainer.from_argparse_args(
        train_config
    )

