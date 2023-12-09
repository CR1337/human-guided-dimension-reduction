import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

def main():
    model = SentenceTransformer("all-mpnet-base-v2")
    dataset = pd.read_csv("imdb.csv")
    dataset["embeddings"] = None
    for i in tqdm(range(len(dataset)), desc=f"writing embeddings"):
        embeddings = model.encode(dataset.iloc[i]["text"])
        dataset.at[i, "embeddings"] = embeddings

    # Filter rows with empty embeddings
    dataset = dataset[dataset["embeddings"].notnull()]
    dataset.to_pickle("imdb_embeddings.pkl")

if __name__ == "__main__":
    main()
