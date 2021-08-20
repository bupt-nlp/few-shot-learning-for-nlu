from __future__ import annotations

from typing import List
from collections import defaultdict
import json

from src.schema import (
    Config,
    FewShotExample
)

from src.dataset_reader.base import DataSetReader

class SnipsDataSetReader(DataSetReader):
    def __init__(self, config: Config):
        self.config = config
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
