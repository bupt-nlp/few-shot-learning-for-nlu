from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Dict

from dataclasses_json import dataclass_json
from transformers import PreTrainedTokenizer
from transformers.models.bert import BertModel


class Metric(Enum):
    L2 = 'l2'
    Cosine = 'Cosine'


class FewShotDataSet(Enum):
    Snips = 'snips'
    Clinc = 'clinc'



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
