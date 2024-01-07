import numpy as np
import pickle
from tqdm import tqdm


with open("./volumes/data/imdb_embeddings.pkl", "rb") as file:
    dataset = pickle.load(file)

dataset = dataset.sort_index()
dataset_length = len(dataset)
print(f"Dataset Length: {dataset_length}")

with open("./volumes/data/imdb_embeddings.bin", "wb") as file:
    dataset_iterator = tqdm(
        dataset.iterrows(),
        total=dataset_length,
        desc="Writing Embeddings"
    )

    for i, row in dataset_iterator:
        embeddings = np.array(row["embeddings"], dtype=np.float32)
        embedding_bytes = embeddings.tobytes()
        file.write(embedding_bytes)
