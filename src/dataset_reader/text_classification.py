from __future__ import annotations

from typing import List
from collections import defaultdict
import json

from src.schema import (
    Config,
    FewShotExample,
    TextClassificationInputExample
)

from src.dataset_reader.base import DataSetReader


class TextClassificationDataSetReader(DataSetReader):
    def read(self, file: str) -> List[TextClassificationInputExample]:
        examples = []
        with open(file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                example: TextClassificationInputExample = TextClassificationInputExample.from_json(line)
                example.example_id = index
                examples.append(example)
        return examples


class SnipsDataSetReader(DataSetReader):
    def __init__(self, config: Config):
        super().__init__(config)
        self.label_domain = defaultdict(str)

    def _get_domain(self, label: str) -> str:
        return self.label_domain[label]

    def read(self, file: str) -> List[FewShotExample]:
        """snips dataset read example from file"""

        examples: List[FewShotExample] = []
        with open(file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                data = json.loads(line)
                examples.append(FewShotExample(
                    id=index,
                    label=data['label'],
                    raw_text=data['text'],
                    domain=self._get_domain(data['label'])
                ))
        return examples


class CLINCDataSetReader(DataSetReader):
    """read CLINC dataset"""
    def __init__(self, config: Config):
        super(CLINCDataSetReader, self).__init__(config)
        self.label_domain = defaultdict(str)

    def _get_domain(self, label_name: str) -> str:
        return self.label_domain[label_name]

    def read(self, file: str) -> List[TextClassificationInputExample]:
        examples = []
        with open(file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                example = TextClassificationInputExample.from_json(line)
                example.example_id = index
                examples.append(example)
        return examples


class AskUbuntuDataSetReader(CLINCDataSetReader):
    pass


class HwuDataSetReader(CLINCDataSetReader):
    pass


class WebApplicationDataSetReader(CLINCDataSetReader):
    pass


class BankingDataSetReader(CLINCDataSetReader):
    pass





