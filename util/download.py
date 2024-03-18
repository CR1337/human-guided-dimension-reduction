import argparse
from datasets import load_dataset

def download_dataset(dataset_name):
    return load_dataset(dataset_name, split=["train"], cache_dir="./volumes/data/cache")

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
        "--num_samples",
        type=int,
        help="Number of samples to download",
        default=2000,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    download_dataset_name = "dair-ai/emotion" if args.dataset_name == "emotion" else args.dataset_name
    dataset = download_dataset(download_dataset_name)[0]
    dataset = dataset.shuffle()
    dataset = dataset.select(range(args.num_samples))
    print(len(dataset))
    save_dataset(dataset, f"./volumes/data/{args.dataset_name}")


if __name__ == "__main__":
    main()
