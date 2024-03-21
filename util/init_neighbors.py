import gc
import os
import sys
import subprocess
from platform import system
from typing import List

BACKEND_PATH: str = os.path.join(os.getcwd(), 'services', 'backend')
sys.path.append(BACKEND_PATH)

from neighbors import Neighbors, ComputedNeighbors  # noqa: E402
from dataset import Dataset  # noqa: E402

NEIGHBORS_EXECUTABLE_PATH: str = os.path.join(
    BACKEND_PATH, "neighbors",
    "neighbors.exe" if system() == "Windows" else "neighbors"
)
COMPILE_SCRIPT_PATH: str = os.path.join(
    BACKEND_PATH, "compile-neighbors"
)

DATASETS: List[Dataset] = Dataset.all()

DIMENSIONS: int = Neighbors.DIMENSIONS_768


if not os.path.exists(NEIGHBORS_EXECUTABLE_PATH):
    print("Compiling neighbors executable...")
    process = subprocess.run(COMPILE_SCRIPT_PATH, shell=True)
    if process.returncode != 0:
        raise RuntimeError("Failed to compile neighbors executable!")

for dataset in DATASETS:
    print(f"\n\nProcessing dataset: {dataset.name}")
    print("Computing euclidean neighbors...")
    euclidean_neighbors = ComputedNeighbors(
        distance_metric="euclidean",
        dimensions=DIMENSIONS,
        dataset=dataset.dataframe
    )
    print("Writing euclidean neighbors to disk...")
    euclidean_neighbors.dump(dataset.euclidean_neighbors_path)
    del euclidean_neighbors
    gc.collect()

    print("Computing cosine neighbors...")
    cosine_neighbors = ComputedNeighbors(
        distance_metric="cosine",
        dimensions=DIMENSIONS,
        dataset=dataset.dataframe
    )
    print("Writing cosine neighbors to disk...")
    cosine_neighbors.dump(dataset.cosine_neighbors_path)
    del cosine_neighbors
    gc.collect()

print("Done!")
