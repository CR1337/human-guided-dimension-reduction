import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from util.knn_old import KNNCalculator


def main():
    model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    dataset = pd.read_csv("./volumes/data/imdb.csv")
    dataset["embeddings"] = None
    for i in tqdm(range(len(dataset)), desc="writing embeddings"):
        embeddings = model.encode(dataset.iloc[i]["text"])
        dataset.at[i, "embeddings"] = embeddings

    # Filter rows with empty embeddings
    dataset = dataset[dataset["embeddings"].notnull()]

    # Compute nearest neighbors
    neighbor_count = 7
    dataset = dataset.reset_index(drop=True)
    knn_calculator = KNNCalculator(dataset, neighbor_count)
    knn_calculator.compute()
    dataset = knn_calculator.result

    dataset.to_pickle("imdb_embeddings.pkl")


if __name__ == "__main__":
    main()
