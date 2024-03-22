import argparse
from datasets import load_dataset


def save_dataset(dataset, dataset_name):
    # Save dataset as a csv file
    dataset.to_csv(dataset_name + ".csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Download a dataset")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to download",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to download",
        default=2000,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset_name == "imdb":
        dataset = load_dataset(
            "imdb", split=["train"], cache_dir="./volumes/data/cache"
        )[0]
    elif args.dataset_name == "emotion":
        dataset = load_dataset(
            "dair-ai/emotion", split=["train"], cache_dir="./volumes/data/cache"
        )[0]
    elif args.dataset_name == "mnli":
        dataset = load_dataset(
            "glue", "mnli", split=["train"], cache_dir="./volumes/data/cache"
        )[0]
        dataset = dataset.map(lambda x: {"text": x["premise"] + ";" + x["hypothesis"]})
        dataset = dataset.remove_columns(["idx", "premise", "hypothesis"])
    else:
        raise ValueError(f"Dataset {args.dataset_name} not found")
    dataset = dataset.shuffle()
    dataset = dataset.select(range(args.num_samples))
    print(len(dataset))
    dataset.to_csv(args.save_dir + ".csv")


if __name__ == "__main__":
    main()
