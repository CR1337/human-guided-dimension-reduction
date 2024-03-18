import glob
import yaml
import os
from pathlib import Path
import csv

import train

CHECKPOINT_DIR = "best_checkpoints"
DATA_DIR = "util/training/data"


def evaluate():
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


def wait_for_debugger(port: int = 56789):
    """
    Pauses the program until a remote debugger is attached.
    Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using "
        f"docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()


if __name__ == "__main__":
    # wait_for_debugger()
    evaluate()
