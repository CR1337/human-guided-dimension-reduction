from pathlib import Path
from simple_parsing import field, list_field
from typing import Literal, List


class TrainingArgs:
    data_dir: Path = field(alias="-d", default="data/imdb")
    model_name: str = field(alias="-m", default="TwoLayerModel")
    epochs: int = field(alias="-e", default=10)
    batch_size: int = field(alias="-b", default=32)
    learning_rate: float = field(alias="-lr", default=1e-3)
    weight_decay: float = field(alias="-wd", default=0.0)
    micro_batch_size: int = field(alias="-mb", default=1)
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "bf16-mixed"
    run_name: str = field(alias="-rn", default="default")
    debug: bool = field(alias="--debug", default=False)

    in_features: int = field(alias="-if", default=90)
    model_params: List[str] = list_field(alias="-mp", default=[32, 16])