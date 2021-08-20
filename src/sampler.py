from __future__ import annotations

from typing import List
from src.schema import FewShotExample


def sample_support_set(examples: List[FewShotExample], n_way: int, k_shot: int, labels: List[str]) -> List[FewShotExample]:
    labels = set()
    