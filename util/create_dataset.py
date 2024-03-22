import argparse
import random
from typing import List, Dict
import os
from os import path
import sys

import jsonlines
from tqdm import tqdm
import numpy as np

sys.path.append(
    path.dirname(path.dirname(path.abspath(__file__))) + "/services/backend"
)
from dr import DimensionalityReduction  # noqa: E402
from dataset import Dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset for training")
    parser.add_argument("data_dir", type=str, help="Directory to store the dataset")
    parser.add_argument(
        "-ts",
        "--train_size",
        type=int,
        default=10_500,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "-vs",
        "--validation_size",
        default=5_250,
        type=float,
        help="Size of the validation set",
    )
    parser.add_argument(
        "-tes", "--test_size", default=5_250, type=float, help="Size of the test set"
    )
    parser.add_argument(
        "-ll",
        "--landmark_lower_bound",
        default=10,
        type=int,
        help="Lower bound for number of landmarks (Including)",
    )
    parser.add_argument(
        "-ul",
        "--landmark_upper_bound",
        default=30,
        type=int,
        help="Upper bound for number of landmarks (Including)",
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
        "-d", "--data_set", type=str, help="Dataset name.", default="imdb_small"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = Dataset(args.data_set, no_neighbors=True)

    # Since in python ranges are excluding the upper bound, we need to add 1
    landmark_upper_bound = args.landmark_upper_bound + 1

    num_samples = args.train_size + args.validation_size + args.test_size
    samples_per_landmark_count = num_samples // (
        landmark_upper_bound - args.landmark_lower_bound
    )
    random.seed(args.seed)
    seeds = random.sample(range(1_000_000), samples_per_landmark_count)

    train_seed_count = int(args.train_size / num_samples * samples_per_landmark_count)
    val_seed_count = int(
        args.validation_size / num_samples * samples_per_landmark_count
    )
    train_seeds = seeds[:train_seed_count]
    val_seeds = seeds[train_seed_count : train_seed_count + val_seed_count]
    test_seeds = seeds[train_seed_count + val_seed_count :]

    train, val, test = [], [], []
    for i in tqdm(
        range(args.landmark_lower_bound, landmark_upper_bound),
        desc="Generating data. For each number of landmarks independently",
    ):
        train += generate_data(i, dataset, train_seeds, args.distance_metric)
        val += generate_data(i, dataset, val_seeds, args.distance_metric)
        test += generate_data(i, dataset, test_seeds, args.distance_metric)

    # This is slow, but it is only done once and a sanity check
    # to make sure that the sets are disjoint
    for train_example in train:
        if train_example in val or train_example in test:
            raise ValueError("Train and validation or test set overlap")

    print("Saving datasets")
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
    num_landmarks: int, dataset: Dataset, seeds: List[int], distance_metric: str
) -> Dict[str, np.ndarray]:
    dr = DimensionalityReduction(
        heuristic="random",
        distance_metric=distance_metric,
        num_landmarks=num_landmarks,
        dataset=dataset,
        create_dataset=True,
    )
    result = []
    for seed in seeds:
        dr.select_landmarks(seed=seed)
        dr.reduce_landmarks()

        original_position = np.vstack(dr.landmarks["embeddings"].apply(np.array))
        projected_position = np.vstack(dr.landmarks["position"].apply(np.array))

        original_distances = dr.distances(original_position, original_position).tolist()
        projected_distances = dr.distances(
            projected_position, projected_position
        ).tolist()

        result.append({"label": original_distances, "input": projected_distances})
    return result


if __name__ == "__main__":
    main()
