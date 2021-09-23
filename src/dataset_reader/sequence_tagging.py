from __future__ import annotations

from typing import List
from collections import defaultdict
import pandas as pd

from src.config import (
    Config
)
from src.schema import SequenceTaggingInputExample

from src.dataset_reader.base import DataSetReader


class SnipsDataSetReader(DataSetReader):
    def __init__(self, config: Config):
        super().__init__(config)
        self.label_domain = defaultdict(str)

    def _get_domain(self, label: str) -> str:
        return self.label_domain[label]

    def read(self, file: str) -> List[SequenceTaggingInputExample]:
        """snips dataset read example from file"""
        table = pd.read_csv(file)
        examples: List[SequenceTaggingInputExample] = []
        for row_index, row in table.iterrows():
            examples.append(SequenceTaggingInputExample(
                example_id=row_index,
                raw_text=row['text'],
                labels=row['slot'].split()
            ))
        return examples


class ATISDataSetReader(SnipsDataSetReader):
    pass

