import gc
import os
import sys
import pickle
import subprocess
from typing import List

BACKEND_PATH: str = os.path.join(os.getcwd(), 'services', 'backend')
NEIGHBORS_EXECUTABLE_PATH: str = os.path.join(
    BACKEND_PATH, "neighbors", "neighbors"
)
COMPILE_SCRIPT_PATH: str = os.path.join(
    BACKEND_PATH, "compile-neighbors"
)

DATASET_TAGS: List[str] = ["imdb_embeddings", "imdb_embeddings_small"]
DATASET_PATHS: List[str] = [
    os.path.join("volumes", "data", "imdb_embeddings.pkl"),
    os.path.join("volumes", "data", "imdb_embeddings_small.pkl"),
]
EUCLIDEAN_OUTPUT_PATHS: List[str] = [
    os.path.join("volumes", "data", "imdb_euclidean_neighbors.bin"),
    os.path.join("volumes", "data", "imdb_euclidean_neighbors_small.bin"),
]
COSINE_OUTPUT_PATHS: List[str] = [
    os.path.join("volumes", "data", "imdb_cosine_neighbors.bin"),
    os.path.join("volumes", "data", "imdb_cosine_neighbors_small.bin"),
]

DIMENSIONS: int = 768

sys.path.append(BACKEND_PATH)

from neighbors import ComputedNeighbors  # noqa: E402

if not os.path.exists(NEIGHBORS_EXECUTABLE_PATH):
    print("Compiling neighbors executable...")
    process = subprocess.run(COMPILE_SCRIPT_PATH, shell=True)
    if process.returncode != 0:
        raise RuntimeError("Failed to compile neighbors executable!")

for (
    dataset_tag, dataset_path, euclidean_output_path, cosine_output_path
) in zip(
    DATASET_TAGS, DATASET_PATHS, EUCLIDEAN_OUTPUT_PATHS, COSINE_OUTPUT_PATHS
):
    print(f"Loading dataset '{dataset_tag}'...")
    with open(dataset_path, "rb") as dataset_file:
        dataset = pickle.load(dataset_file)

    print(dataset)
    if "small" not in dataset_tag:
        continue

    k = len(dataset) - 1

    print("Computing euclidean neighbors...")
    euclidean_neighbors = ComputedNeighbors(
        distance_metric="euclidean",
        k=k,
        dimensions=DIMENSIONS,
        dataset=dataset
    )
    print("Writing euclidean neighbors to disk...")
    euclidean_neighbors.dump(euclidean_output_path)
    del euclidean_neighbors
    gc.collect()

    print("Computing cosine neighbors...")
    cosine_neighbors = ComputedNeighbors(
        distance_metric="cosine",
        k=k,
        dimensions=DIMENSIONS,
        dataset=dataset
    )
    print("Writing cosine neighbors to disk...")
    cosine_neighbors.dump(cosine_output_path)
    del cosine_neighbors
    gc.collect()

print("Done!")
