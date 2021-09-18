from __future__ import annotations

from typing import List, Dict
import random
from collections import defaultdict

from src.schema import FewShotExample, Config
from src.config import SEED

random.seed(SEED)


class Sampler:
    def __init__(self, config: Config, examples: List[FewShotExample]):
        self.config = config
        self.examples = examples

    def sample_shuffle_data(self, labels: List[str]) -> Dict[str, List[FewShotExample]]:
        """
        sample few-shot examples with random shuffle strategy
        Args:
            labels: the target n-way labels

        Returns: the label few-shot examples
        """

        # 1. construct label examples and shuffle them
        label_examples = defaultdict(list)
        for example in self.examples:
            label_examples[example.label].append(example)

        for label in label_examples.keys():
            random.shuffle(label_examples[label])

        # 2. choose the top k labels
        support_set = defaultdict(list)
        for label in labels:
            assert label in label_examples
            support_set[label] = label_examples[label][: self.config.k_shot_train]

        return support_set

    def get_train_examples(self, label_examples: Dict[str, List[FewShotExample]]) -> List[FewShotExample]:
        """
        get train examples exclude by label support set
        Args:
            label_examples:

        Returns:

        """
        example_ids = set()
        for examples in label_examples.values():
            for example in examples:
                example_ids.add(example.id)

        train_examples: List[FewShotExample] = []
        for example in self.examples:
            if example.id in example_ids:
                continue
            train_examples.append(example)

        return train_examples
