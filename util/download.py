from datasets import load_dataset


def download(csv_filename: str, name: str):
    dataset = load_dataset(
        name, split=['train'], cache_dir="./volumes/data/cache"
    )[0]
    dataset.to_csv(csv_filename)
    return dataset


if __name__ == "__main__":
    # Change args before running:
    dataset = download("./volumes/data/imdb.csv", "imdb")
    print(len(dataset))
