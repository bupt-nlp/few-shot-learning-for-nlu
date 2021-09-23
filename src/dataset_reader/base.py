"""base module for dataset reader"""
from __future__ import annotations

from src.config import (
    Config
)


class DataSetReader:
    """base DataSet reader"""
    def __init__(self, config: Config):
        self.config = config

    def read(self, file: str):
        """read example from file"""
        raise NotImplementedError
