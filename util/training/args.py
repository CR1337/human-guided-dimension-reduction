from pathlib import Path
from simple_parsing import field, list_field
from typing import Literal, List


class TrainingArgs:
    # Run parameters
    accelerator: Literal["cpu", "cuda"] = "cuda"
    micro_batch_size: int = field(alias="-mb", default=1)
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "bf16-mixed"
    run_name: str = field(alias="-rn", default="default")
    debug: bool = field(alias="--debug", default=False)

    # Hyper parameters
    epochs: int = field(alias="-e", default=10)
    batch_size: int = field(alias="-b", default=32)
    learning_rate: float = field(alias="-lr", default=1e-3)
    weight_decay: float = field(alias="-wd", default=0.0)

    # Data parameters
    data_dir: Path = field(alias="-d", default="data/imdb")
    train_file: str = field(default="train.jsonl")
    val_file: str = field(default="dev.jsonl")
    test_file: str = field(default="test.jsonl")

    # Model parameters
    model_name: str = field(alias="-m", default="TwoLayerModel")
    max_landmarks: int = field(alias="-ml", default=30)
    model_params: List[str] = list_field(alias="-mp", default=[32, 16])