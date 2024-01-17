import gc
import os
import sys
import pickle
import subprocess

BACKEND_PATH: str = os.path.join(os.getcwd(), 'services', 'backend')
NEIGHBORS_EXECUTABLE_PATH: str = os.path.join(
    BACKEND_PATH, "neighbors", "neighbors"
)
COMPILE_SCRIPT_PATH: str = os.path.join(
    BACKEND_PATH, "compile-neighbors"
)
EUCLIDEAN_OUTPUT_PATH: str = os.path.join(
    "volumes", "data", "imdb_euclidean_neighbors.bin"
)
COSINE_OUTPUT_PATH: str = os.path.join(
    "volumes", "data", "imdb_cosine_neighbors.bin"
)
DATASET_PATH: str = os.path.join(
    "volumes", "data", "imdb_embeddings.pkl"
)

DIMENSIONS: int = 768

sys.path.append(BACKEND_PATH)

from neighbors import ComputedNeighbors  # noqa: E402

if not os.path.exists(NEIGHBORS_EXECUTABLE_PATH):
    print("Compiling neighbors executable...")
    process = subprocess.run(COMPILE_SCRIPT_PATH, shell=True)
    if process.returncode != 0:
        raise RuntimeError("Failed to compile neighbors executable!")

print("Loading dataset...")
with open(DATASET_PATH, "rb") as dataset_file:
    dataset = pickle.load(dataset_file)

k = len(dataset) - 1

print("Computing euclidean neighbors...")
euclidean_neighbors = ComputedNeighbors(
    distance_metric="euclidean",
    k=k,
    dimensions=DIMENSIONS,
    dataset=dataset
)
print("Writing euclidean neighbors to disk...")
euclidean_neighbors.dump(EUCLIDEAN_OUTPUT_PATH)
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
cosine_neighbors.dump(COSINE_OUTPUT_PATH)
del cosine_neighbors
gc.collect()

print("Done!")
