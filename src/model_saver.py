import json
import uuid
from pathlib import Path

import torch
import torch.nn as nn


class ModelSaver:
    def __init__(self, index_path: Path, state_dicts_path: Path) -> None:
        self.index_path = index_path
        if not self.index_path.exists():
            self.index_path.touch()
        self.state_dicts_path = state_dicts_path
        if not self.state_dicts_path.is_dir():
            self.state_dicts_path.mkdir()
        return

    def save_model(self, model: nn.Module, **kwargs) -> None:
        model_name = uuid.uuid4().__str__()
        torch.save(model.state_dict(), self.state_dicts_path / f"{model_name}.pth")
        with open(self.index_path, "w") as index_file:
            json.dump({"model_name": model_name, **kwargs}, index_file)
        return
