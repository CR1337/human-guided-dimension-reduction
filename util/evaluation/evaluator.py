import pickle
import argparse
import numpy as np

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
    parser.add_argument("-mp", "--model_path", type=str, default="checkpoints/OneModel_best")
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
        model_path=args.model_path
    )
    metrics = []
    for seed in SEEDS:
        lmds.select_landmarks(seed=seed)
        lmds.reduce_landmarks()
        lmds.low_landmark_embeddings = move_landmarks(lmds.landmarks["label"])
        lmds.calculate(imds_algorithm=args.imds_algorithm, do_pca=False)
        metrics.append(lmds.compute_metrics(7))

    print(f"Mean metrics for {args.imds_algorithm} with {args.landmark_count} landmarks")
    print(np.mean(metrics, axis=0))
    print(np.std(metrics, axis=0))

def move_landmarks(landmark_labels):
    landmark_positions = []
    for label in landmark_labels:
        if label == 0:
            landmark_positions.append([1,1])
        else:
            landmark_positions.append([-1,-1])
    return np.array(landmark_positions)


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
    main()

