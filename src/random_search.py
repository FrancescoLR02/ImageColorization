from typing import Any, Dict, List
import random


class RandomPicker:
    def __init__(self, param_space: Dict[str, List[Any]]) -> None:
        self.param_space = param_space
        return

    def pick(self) -> Dict[str, Any]:
        param_sampled = {}
        for key, value_list in self.param_space.items():
            param_sampled[key] = random.choice(value_list)
        return param_sampled
