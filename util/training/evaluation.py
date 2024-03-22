import glob
import yaml
import os
from pathlib import Path
import csv

import train

# Needs to be adjusted based on the individual setup
CHECKPOINT_DIR = "best_checkpoints"
DATA_DIR = "util/training/data"


def evaluate():
    """
    Evaluates all checkpoints on all datasets. The results are written to a csv file. This script is to replicate the results from table 2 of the poster.
    """
    checkpoints = glob.glob(f"{CHECKPOINT_DIR}/*")
    datasets = glob.glob(f"{DATA_DIR}/*")
    losses = {}
    print(f"Testing: {checkpoints} on {datasets}")
    for checkpoint in checkpoints:
        print(f"Running evaluation for: {checkpoint}")
        for dataset in datasets:
            params = yaml.safe_load(open(os.path.join(checkpoint, "params.yml")))
            eval_args = {
                "offline": True,
                "only_test": True,
                "load_model": checkpoint,
                "data_dir": Path(dataset),
                "model_param1": params["model_param1"],
                "model_param2": params["model_param2"],
                "inner_activation": params["inner_activation"],
                "end_activation": params["end_activation"],
                "max_landmarks": params["max_landmarks"],
            }
            losses[checkpoint, dataset] = train.main(
                is_evaluation=True, eval_args=eval_args
            )

    with open("evaluation.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "dataset", "loss"])
        for (checkpoint, dataset), loss in losses.items():
            writer.writerow([checkpoint, dataset, loss])


if __name__ == "__main__":
    evaluate()
