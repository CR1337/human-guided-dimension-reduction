import argparse

import pickle
import random
import pandas as pd
from typing import List, Tuple
import numpy as np

from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))) + "/services/backend")
# We need to import metrics and neighbors to allow for the import of Lmds
from lmds import Lmds  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser(description='Create dataset for training')
    parser.add_argument("-d", '--data_dir', type=str, help='Directory to store the dataset')
    parser.add_argument("-n", '--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument("-s", "--split", type=float, default=0.8, help="Train-test split")
    parser.add_argument("-ll", "--landmark_lower_bound", default=10, type=int, help="Lower bound for number of landmarks")
    parser.add_argument("-ul", "--landmark_upper_bound", default=100, type=int, help="Upper bound for number of landmarks")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()

    with open("./volumes/data/imdb_embeddings_small.pkl", "rb") as file:
        dataset = pickle.load(file)


    samples_per_landmark_count = args.num_samples // (args.landmark_upper_bound - args.landmark_lower_bound)
    random.seed(args.seed)
    seeds = random.sample(range(100000), samples_per_landmark_count)
    data = []
    for i in range(args.landmark_lower_bound, args.landmark_upper_bound):
        data += generate_data(i, dataset, seeds)
    random.shuffle(data)
    split = int(args.split * len(data))
    train = data[:split]
    test = data[split:]
    with open(args.data_dir + "/train.pkl", "w+") as file:
        pickle.dump(train, file)
    with open(args.data_dir + "/test.csv", "w+") as file:
        pickle.dump(test, file)


def generate_data(num_landmarks: int, dataset: pd.DataFrame, seeds: List[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    lmds = Lmds(
        heuristic="random",
        distance_metric="euclidean",
        num_landmarks=num_landmarks,
        dataset=dataset,
        do_pca=False,
        debug=True,
    )
    result = []
    for seed in seeds:
        lmds.select_landmarks(seed=seed)
        lmds.reduce_landmarks()
        original_position = np.vstack(
            lmds.landmarks["embeddings"].apply(np.array)
        )
        projected_position = np.vstack(
            lmds.landmarks["position"].apply(np.array)
        )
        result.append((
            original_position,
            projected_position
        ))
    return result

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


