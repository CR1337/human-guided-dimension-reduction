from datasets import load_dataset


def download_dataset(dataset_name):
    return load_dataset(
        dataset_name, split=["train"], cache_dir="./volumes/data/cache"
    )


def save_dataset(dataset, dataset_name):
    # Save dataset as a csv file
    dataset.to_csv(dataset_name + ".csv")


def main():
    dataset_name = "imdb"
    dataset = download_dataset(dataset_name)[0]
    print(len(dataset))
    save_dataset(dataset, f"./volumes/data/{dataset_name}")


if __name__ == "__main__":
    main()
