from pathlib import Path
from simple_parsing import field, list_field
from dataclasses import dataclass
from typing import Literal, List


@dataclass(kw_only=True)
class TrainingArgs:
    # Run parameters
    accelerator: Literal["cpu", "cuda"] = "cuda"
    precision: Literal["32-true", "16-mixed", "bf16-mixed", "64-true"] = "64-true"
    run_name: str = field(alias="-rn", default="default")
    debug: bool = field(alias="--debug", default=False)
    seed: int = field(alias="-s", default=42)

    # Hyper parameters
    epochs: int = field(alias="-e", default=10)
    learning_rate: float = field(alias="-lr", default=1e-3)
    weight_decay: float = field(alias="-wd", default=0.0)

    # Data parameters
    data_dir: Path = field(alias="-d", default="util/training/data/test")
    train_file: str = field(default="train.jsonl")
    val_file: str = field(default="val.jsonl")
    test_file: str = field(default="test.jsonl")

    # Model parameters
    model_name: str = field(alias="-m", default="TwoLayerModel")
    max_landmarks: int = field(alias="-ml", default=30)
    model_params: List[str] = list_field(alias="-mp", default=[32, 16])

    def update_from_dict(self, values_dict):
            # Update class variables with values from the dictionary
            for key, value in values_dict.items():
                setattr(self, key, value)