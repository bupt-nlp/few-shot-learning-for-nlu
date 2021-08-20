from __future__ import annotations

from typing import Union, Optional, List, Dict
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

from torch import Tensor
from tap import Tap
from enum import Enum


class Distance(Enum):
    L2 = 'l2'
    Cosine = 'Cosine'
    
    
class FewShotDataSet(Enum):
    Snips = 'snips'


class Config(Tap):
    dataset: FewShotDataSet    # snips
    distance: Distance      # distance
    n_way_train: int            # numbner of support examples per class for training tasks
    n_way_validation: int       # numbner of support examples per class for validation task
    
    k_shot_train: int            # number of classes for training tasks
    k_shot_validation: int       # number of classes for validation tasks
    

class TrainConfig(Tap):
    epoch: int
    steps: int          # steps per epoch
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
        