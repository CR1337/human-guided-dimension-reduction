from pathlib import Path
from simple_parsing import field, list_field
from dataclasses import dataclass
from typing import Literal


@dataclass()
class TrainingArgs:
    # Run parameters
    accelerator: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["32-true", "16-mixed", "bf16-mixed", "64-true"] = "32-true"
    run_name: str = field(alias="-rn", default="default")
    debug: bool = field(alias="--debug", default=False)
    offline: bool = field(alias="--offline", default=False)
    seed: int = field(alias="-s", default=42)
    num_workers: int = field(alias="-nw", default=0)
    only_test: bool = field(alias="--only-test", default=False)
    load_model: str = field(alias="-lm", default=None)
    early_stopping_patience: int = field(alias="-esp", default=5)

    # Hyper parameters
    epochs: int = field(alias="-e", default=10)
    learning_rate: float = field(alias="-lr", default=1e-3)
    beta1: float = field(alias="-b1", default=0.9)
    beta2: float = field(alias="-b2", default=0.999)
    epsilon: float = field(alias="-eps", default=1e-8)
    dropout_prob: float = field(alias="-dp", default=0.2)
    batch_size: int = field(alias="-bs", default=32)

    # Data parameters
    data_dir: Path = field(alias="-d", default="util/training/data/base_set")
    train_file: str = field(default="train.jsonl")
    val_file: str = field(default="val.jsonl")
    test_file: str = field(default="test.jsonl")

    # Model parameters
    model_name: str = field(alias="-m", default="OneLayerModel")
    max_landmarks: int = field(alias="-ml", default=30)
    model_param1: int = list_field(alias="-mp1", default=32)
    model_param2: int = list_field(alias="-mp2", default=0)
    inner_activation: Literal["relu", "tanh", "sigmoid", "identity"] = field(
        default="relu"
    )
    end_activation: Literal["relu", "tanh", "sigmoid", "identity"] = field(
        default="relu"
    )

    def update_from_dict(self, values_dict):
        # Update class variables with values from the dictionary
        for key, value in values_dict.items():
            setattr(self, key, value)
