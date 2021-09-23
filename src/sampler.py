from __future__ import annotations

from typing import List, Dict, Tuple
import random
import os, json
from pprint import pprint
from collections import defaultdict
from dataclasses_json import dataclass_json
from dataclasses import dataclass

from src.schema import FewShotExample, TextClassificationInputExample
from src.dataset_reader.text_classification import CLINCDataSetReader
from src.config import SEED, Config, root_dir, get_logger

from random import Random


@dataclass_json
@dataclass
class Config:
    seed: int = 200
    n_way: int = 5
    k_shot: int = 5


config = Config()
logger = get_logger()


def shuffle(items: list):
    Random(config.seed).shuffle(items)


def sampler_support_set(examples: List[TextClassificationInputExample]) -> Tuple[list, list]:
    labels = set()
    for example in examples:
        labels.add(example.label)

    support_set_labels, train_set_labels = labels[: config.n_way], labels[config.n_way:]

    support_set_indexes = [index for index, example in enumerate(examples) if example.label in support_set_labels]

    shuffle(examples)


def sample_clinc_data(file: str, domain_file: str):
    
    logger.info('sample clinc few-shot dataset')
    reader = CLINCDataSetReader({})
    examples: List[TextClassificationInputExample] = reader.read(file)

    # 1. sampler domains
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_labels = json.load(f)

    label2domain: dict = {}
    for domain_name, labels in domain_labels.items():
        for label in labels:
            label2domain[label] = domain_name

    domain_names = list(domain_labels.keys())
    # 2. sample all domain examples
    for test_index in range(len(domain_names)):
        test_domain_name = domain_names.pop(test_index)

        for val_index in range(len(domain_names)):
            val_domain_name = domain_names.pop(val_index)

            train_examples, val_examples, test_examples = [], [], []

            for example in examples:
                if example.label not in label2domain:
                    raise ValueError(f'error example: {example.raw_text}')
                domain_name = label2domain[example.label]

                if domain_name == test_domain_name:
                    test_examples.append(example)
                elif domain_name == val_domain_name:
                    val_examples.append(example)
                else:
                    train_examples.append(example)

            # 3. sample the n-way k-shot example to train examples
            all_test_labels = [domain_name == test_domain_name for label, domain_name in
                           label2domain.items() if test_domain_name]

            shuffle(all_test_labels)
            # 4. select test labels into
            train_test_labels = all_test_labels[n_way:]

            index = 0
            while index < len(test_examples):
                example = test_examples[index]
                if example.label in train_test_labels:
                    train_examples.append(example)
                    test_examples.pop(index)
                else:
                    index += 1

            # 5. save train, validation, test examples into sub dir


def main():
    intent_corpus_file = os.path.join(root_dir, 'data/clinc/intent-classification/clinc150.json.corpus')
    corpus_dir = os.path.dirname(intent_corpus_file)
    domain_file = os.path.join(corpus_dir, 'domains.json')

    sample_clinc_data(
        intent_corpus_file,
        domain_file
    )


if __name__ == '__main__':
    main()
