import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def embedd(csv_filename: str, name: str):
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    dataset = pd.read_csv(csv_filename)
    dataset["embeddings"] = None
    for i in tqdm(range(len(dataset)), desc="writing embeddings"):
        embeddings = model.encode(dataset.iloc[i]["text"])
        dataset.at[i, "embeddings"] = embeddings

    # Filter rows with empty embeddings
    dataset = dataset[dataset["embeddings"].notnull()]

    dataset.to_pickle(f"{name}_embeddings.pkl")


if __name__ == "__main__":
    # Change args before running:
    embedd("./volumes/data/glue_mnli.csv", "gluie_mnli")
