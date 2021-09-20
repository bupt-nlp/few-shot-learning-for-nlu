from typing import List
from torch import Tensor
from pytest import fixture
from transformers import BatchEncoding
from transformers.models.bert import BertForSequenceClassification, BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


@fixture
def model_name():
    return 'bert-base-uncased'


@fixture
def tokenizer(model_name) -> BertTokenizer:
    return BertTokenizer.from_pretrained(model_name)


@fixture
def classifier(model_name) -> BertForSequenceClassification:
    bert_for_sequence_classification = BertForSequenceClassification.from_pretrained(model_name)
    return bert_for_sequence_classification


@fixture
def bert_model(model_name) -> BertModel:
    return BertModel.from_pretrained(model_name)


@fixture
def sentence() -> List[str]:
    return ["I love china", "I hate china"]


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


@fixture
def batch(tokenizer: BertTokenizer, sentence: List[str]):
    inputs = tokenizer(
        sentence,
        padding=True,
        return_tensors='pt',
        return_token_type_ids=True,
        return_attention_mask=True,
        max_length=100
    )
    return inputs


