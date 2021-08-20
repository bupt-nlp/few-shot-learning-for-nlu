from __future__ import annotations

from typing import List
from src.schema import (
    Config,
    FewShotExample
)

class DataSetReader:
    """base DataSet reader"""
    def __init__(self, config: Config):
        self.config = config
        
    def read(self, file: str) -> List[FewShotExample]:
        """read example from file"""
        raise NotImplementedError
