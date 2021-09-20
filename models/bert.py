from __future__ import annotations

from typing import Optional
from torch import Tensor, optim
from transformers.models.bert import BertForSequenceClassification

import pytorch_lightning as pl

from src.schema import Config, TrainConfig


class BertForIntentDetection(pl.LightningModule):
    def __init__(self, config: Config, train_config: TrainConfig):
        super(BertForIntentDetection, self).__init__()
        self.config = config
        self.train_config = train_config
        self.classifier = BertForSequenceClassification.from_pretrained(config.pretrain_model)

    def training_step(self, batch, batch_index) -> Tensor:
        return self.classifier(*batch)

    def validation_step(self, batch, batch_index) -> Tensor:
        return self.classifier(*batch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config)