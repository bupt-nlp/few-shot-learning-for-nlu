from __future__ import annotations

from typing import Optional

import setuptools.config
from torch import Tensor, optim
from transformers import BertConfig
from transformers.models.bert import BertForSequenceClassification, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

import pytorch_lightning as pl

from src.config import Config, TrainerConfig
from src.schema import Metric
from src.utils import get_metric


class BertForIntentDetection(pl.LightningModule):
    def __init__(self, config: Config, train_config: TrainerConfig):
        super(BertForIntentDetection, self).__init__()
        self.config = config
        self.train_config = train_config

        # 1. init the bert model
        bert_config = BertConfig.from_pretrained(config.pretrain_model)
        self.encoder = BertModel(bert_config)

    def get_embedding(self, input_ids, attention_mask, token_type_ids) -> Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Tensor:
        support_num = self.config.k_shot * self.config.n_way

        support_set_input_ids, support_set_attention_mask = input_ids[:support_num], attention_mask[:support_num]
        support_set_token_type_ids = token_type_ids[:support_num]

        query_set_input_ids, query_set_attention_mask = input_ids[support_num:], attention_mask[support_num:]
        query_set_token_type_ids = token_type_ids[support_num:]

        support_set = self.get_embedding(
            support_set_input_ids,
            support_set_attention_mask,
            support_set_token_type_ids
        )
        query_set = self.get_embedding(
            query_set_input_ids,
            query_set_attention_mask,
            query_set_token_type_ids
        )
        metric = get_metric(query_set, support_set, self.config.metric)
        return metric

    def training_step(self, batch, batch_index) -> Tensor:
        input_ids, attention_mask, token_type_ids = batch
        loss = self.forward(input_ids, attention_mask, token_type_ids)
        return loss

    def validation_step(self, batch, batch_index) -> Tensor:
        input_ids, attention_mask, token_type_ids = batch
        loss = self.forward(input_ids, attention_mask, token_type_ids)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.learning_rate)
        return optimizer
