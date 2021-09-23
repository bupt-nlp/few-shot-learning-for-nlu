from __future__ import annotations

from typing import Optional
from torch import Tensor, optim
from transformers import BertConfig
from transformers.models.bert import BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import pytorch_lightning as pl

from src.config import Config, TrainerConfig


class ProtoNetworkForIntentDetection(pl.LightningModule):
    def __init__(self, config: Config, train_config: TrainerConfig):
        super(ProtoNetworkForIntentDetection, self).__init__()
        self.config = config
        self.train_config = train_config

        # 1. init the bert model
        bert_config = BertConfig.from_pretrained(config.pretrain_model)
        bert_config.num_labels = config.num_labels
        self.classifier = BertForSequenceClassification(bert_config)

    def forward(self, input_ids, attention_mask, token_type_ids, labels) -> SequenceClassifierOutput:
        return self.classifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

    def training_step(self, batch, batch_index) -> Tensor:
        input_ids, attention_mask, token_type_ids, labels = batch
        output = self.forward(input_ids, attention_mask, token_type_ids, labels)
        return output.loss

    def validation_step(self, batch, batch_index) -> Tensor:
        input_ids, attention_mask, token_type_ids, labels = batch
        output = self.forward(input_ids, attention_mask, token_type_ids, labels)
        return output.logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.learning_rate)
        return optimizer
