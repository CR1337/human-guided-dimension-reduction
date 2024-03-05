import pickle
import argparse

import sys
from os import path

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.abspath(__file__)))) + "/services/backend"
)
from lmds import Lmds  # noqa: E402

SEEDS = [42, 642465, 87575675]
LANDMARK_NUMS = [10, 20, 30]

def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset for training")
    parser.add_argument("imds_algorithm", type=str)
    parser.add_argument(
        "landmark_count",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open("./volumes/data/imdb_embeddings_small.pkl", "rb") as file:
        dataset = pickle.load(file)

    lmds = Lmds(
        heuristic="random",
        distance_metric="euclidean",
        num_landmarks=args.landmark_count,
        dataset=dataset,
        debug=True,
    )
    for seed in SEEDS:
        lmds.select_landmarks(seed=seed)
        lmds.reduce_landmarks()
        lmds.calculate(imds_algorithm=args.imds_algorithm, do_pca=False)
        metrics = lmds.compute_metrics(7)
        print(metrics)


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
    wait_for_debugger()
    main()

