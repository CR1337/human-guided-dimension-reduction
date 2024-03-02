import argparse

import pickle
import random
import pandas as pd
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import os
import jsonlines

from os import path
import sys

sys.path.append(
    path.dirname(path.dirname(path.abspath(__file__))) + "/services/backend"
)
# We need to import metrics and neighbors to allow for the import of Lmds
from lmds import Lmds  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset for training")
    parser.add_argument("data_dir", type=str, help="Directory to store the dataset")
    parser.add_argument(
        "-ts",
        "--train_size",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "-vs",
        "--validation_size",
        default=500,
        type=float,
        help="Size of the validation set",
    )
    parser.add_argument(
        "-tes", "--test_size", default=500, type=float, help="Size of the test set"
    )
    parser.add_argument(
        "-ll",
        "--landmark_lower_bound",
        default=10,
        type=int,
        help="Lower bound for number of landmarks",
    )
    parser.add_argument(
        "-ul",
        "--landmark_upper_bound",
        default=30,
        type=int,
        help="Upper bound for number of landmarks",
    )
    parser.add_argument(
        "-dm",
        "--distance_metric",
        default="euclidean",
        type=str,
        help="The metric to measure distances with",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--debug", action="store_true", help="Wait for debugger to attach"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        wait_for_debugger()

    with open("./volumes/data/imdb_embeddings_small.pkl", "rb") as file:
        dataset = pickle.load(file)

    num_samples = args.train_size + args.validation_size + args.test_size
    samples_per_landmark_count = num_samples // (
        args.landmark_upper_bound - args.landmark_lower_bound
    )
    random.seed(args.seed)
    seeds = random.sample(range(100000), samples_per_landmark_count)

    train_seed_count = int(args.train_size / num_samples * samples_per_landmark_count)
    val_seed_count = int(
        args.validation_size / num_samples * samples_per_landmark_count
    )
    train_seeds = seeds[:train_seed_count]
    val_seeds = seeds[train_seed_count : train_seed_count + val_seed_count]
    test_seeds = seeds[train_seed_count + val_seed_count :]

    train, val, test = [], [], []
    # TODO: Off by one error?
    for i in tqdm(
        range(args.landmark_lower_bound, args.landmark_upper_bound),
        desc="Generating data. For each number of landmarks independently",
    ):
        train += generate_data(i, dataset, train_seeds, args.distance_metric)
        val += generate_data(i, dataset, val_seeds, args.distance_metric)
        test += generate_data(i, dataset, test_seeds, args.distance_metric)

    os.makedirs(args.data_dir, exist_ok=True)
    with open(args.data_dir + "/train.jsonl", "wb+") as file:
        with jsonlines.Writer(file, compact=True) as writer:
            writer.write_all(train)
    with open(args.data_dir + "/val.jsonl", "wb+") as file:
        with jsonlines.Writer(file, compact=True) as writer:
            writer.write_all(val)
    with open(args.data_dir + "/test.jsonl", "wb+") as file:
        with jsonlines.Writer(file, compact=True) as writer:
            writer.write_all(test)


def generate_data(
    num_landmarks: int, dataset: pd.DataFrame, seeds: List[int], distance_metric: str
) -> Dict[str, np.ndarray]:
    lmds = Lmds(
        heuristic="random",
        distance_metric=distance_metric,
        num_landmarks=num_landmarks,
        dataset=dataset,
        create_dataset=True,
    )
    result = []
    for seed in seeds:
        lmds.select_landmarks(seed=seed)
        lmds.reduce_landmarks()

        original_position = np.vstack(lmds.landmarks["embeddings"].apply(np.array))
        projected_position = np.vstack(lmds.landmarks["position"].apply(np.array))

        original_distances = lmds.distances(
            original_position, original_position
        ).tolist()
        projected_distances = lmds.distances(
            projected_position, projected_position
        ).tolist()

        result.append({"label": original_distances, "input": projected_distances})
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
    main()
