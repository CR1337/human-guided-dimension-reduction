from datasets import load_dataset

def download_dataset(dataset_name):
    dataset = load_dataset(dataset_name, split=["train"], cache_dir="~/datasets")
    return dataset

def save_dataset(dataset, dataset_name):
    #Save dataset as a csv file
    dataset.to_csv(dataset_name + ".csv")


def main():
    dataset = download_dataset("imdb")[0]
    print(len(dataset))
    save_dataset(dataset, "imdb")

if __name__ == "__main__":
    main()