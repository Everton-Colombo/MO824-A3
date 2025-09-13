from dataclasses import dataclass
from typing import List, Set

@dataclass
class ScQbfSolution:
    elements: List[int]
    _objfun_val: float = None