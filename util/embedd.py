import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def arg_parser():
    parser = argparse.ArgumentParser(description="Download a dataset")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset to embed",
    )
    return parser.parse_args()


def main():
    args = arg_parser()
    model = SentenceTransformer("all-mpnet-base-v2")
    dataset = pd.read_csv(f"./volumes/data/{args.dataset_name}.csv")
    dataset["embeddings"] = None
    for i in tqdm(range(len(dataset)), desc="writing embeddings"):
        embeddings = model.encode(dataset.iloc[i]["text"])
        dataset.at[i, "embeddings"] = embeddings

    # Filter rows with empty embeddings
    dataset = dataset[dataset["embeddings"].notnull()]

    dataset.to_pickle(f"volumes/data/{args.dataset_name}_embeddings.pkl")


if __name__ == "__main__":
    main()
