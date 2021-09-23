from typing import List
from torch import Tensor, LongTensor
from pytest import fixture
from transformers import BatchEncoding, BertConfig
from transformers.models.bert import BertForSequenceClassification, BertTokenizer, BertModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput
)


@fixture
def model_name():
    return 'bert-base-uncased'


@fixture
def num_labels() -> int:
    return 3


@fixture
def tokenizer(model_name) -> BertTokenizer:
    return BertTokenizer.from_pretrained(model_name)


@fixture
def classifier(model_name, num_labels) -> BertForSequenceClassification:
    bert_config = BertConfig.from_pretrained(model_name)
    bert_config.num_labels = num_labels
    bert_for_sequence_classification = BertForSequenceClassification(bert_config)
    return bert_for_sequence_classification


@fixture
def bert_model(model_name, num_labels) -> BertModel:
    bert_config = BertConfig.from_pretrained(model_name)
    bert_config.num_labels = num_labels
    return BertModel(bert_config)


@fixture
def sentence() -> List[str]:
    return ["I love china", "I hate china"]


@fixture
def labels() -> List[int]:
    return [1, 0]


def test_tokenizer_call(tokenizer: BertTokenizer, sentence: List[str]):
    # 可针对于single text or batch text 进行编码
    first_sentence, second_sentence = ["I love china", "I hate china"], ["I also love china", "I also hate china"]
    output: BatchEncoding = tokenizer.__call__(
        text=first_sentence,
        text_pair=second_sentence,
        padding=True,
        max_length=20,
        return_tensors='pt',
        return_token_type_ids=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_length=True
    )
    assert output.input_ids.shape == (2, 10)


def test_tokenizer_batch_encode(tokenizer: BertTokenizer, sentence: List[str]):
    first_sentence, second_sentence = ["I love china", "I hate china"], ["I also love china", "I also hate china"]
    output: BatchEncoding = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=first_sentence,
        padding="max_length",
        max_length=200,
        return_tensors='pt',
        return_token_type_ids=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_length=True
    )
    assert output.input_ids.shape == (2, 200)


def test_bert_for_text_classification(tokenizer: BertTokenizer, sentence: List[str], classifier: BertForSequenceClassification, labels: List[int]):
    """test for text classification"""
    inputs = tokenizer(
        sentence,
        padding='max_length',
        return_tensors='pt',
        return_token_type_ids=True,
        return_attention_mask=True,
        max_length=100
    )
    labels = LongTensor(labels)
    output: SequenceClassifierOutput = classifier(
        labels=labels,
        **inputs
    )
    assert output.logits.shape == (2, 3)


