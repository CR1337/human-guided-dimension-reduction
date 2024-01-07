import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import numpy as np


def wait_for_debugger(port: int = 56789):
    """
    Pauses the program until a remote debugger is attached.
    Should only be called on rank0.
    """

    import debugpy

    debugpy.listen(("0.0.0.0", port))
    print(
        f"Waiting for client to attach on port {port}... NOTE: if using "
        f"docker, you need to forward the port with -p {port}:{port}."
    )
    debugpy.wait_for_client()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heuristic",
        type=str,
        default="random",
        help="Heuristic to use for selecting landmarks.",
    )
    parser.add_argument(
        "--num_landmarks",
        type=int,
        default=100,
        help="Number of landmarks to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imdb_embeddings.pkl",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Dimension of the embedding.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Wait for debugger to attach.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lmds.json",
        help="Path to the output file. Supports .pkl and .json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug:
        wait_for_debugger()
    dataset = pd.read_pickle(args.dataset)
    dataset = lmds(args.heuristic, args.num_landmarks, dataset)
    output_type = args.output.split(".")[-1]
    if output_type == "json":
        dataset.to_json(args.output)
    elif output_type == "pkl":
        dataset.to_pickle(args.output)
    else:
        raise NotImplementedError(f"Unknown output type: {output_type}")


def lmds(
    heuristic: str,
    num_landmarks: int,
    dataset: pd.DataFrame,
    dimension: int = 2
):
    """
    Performs LMDS on the dataset.
    This code is inspired from the LMDS paper: http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf and https://github.com/thomasweighill/landmarkMDS/blob/master/LMDS.py
    1. Select Landmarks
    2. Perform MDS on the landmarks
    3. Apply distance-based triangulation
    (4. Recenter the data and apply PCA)
    """
    # 1. Select Landmarks
    landmarks = None
    if heuristic == "random":
        landmarks = dataset.sample(n=num_landmarks)
    else:
        raise NotImplementedError(f"Unknown heuristic: {heuristic}")

    # 2. Perform MDS on the landmarks

    # We need to transform landmarks["embeddings"] into a numpy array
    # with the shape (num_landmarks, embedding_dim)
    landmark_embeddings = np.vstack(landmarks['embeddings'].apply(np.array))

    # Deltan is the distance matrix between the landmarks
    delta_n = euclidean_distances(landmark_embeddings, landmark_embeddings)
    # H is the mean centering matrix
    H = delta_n - 1/num_landmarks
    # B is the mean centered "inner-product" matrix
    B = -1/2 * H @ delta_n @ H
    # We compute the eigenvalues and eigenvectors of B
    eigenvalues, eigenvectors = np.linalg.eig(B)
    # We sort the eigenvalues and eigenvectors by decreasing eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # We compute the matrix L which is given
    # by eigenvectors * sqrt(eigenvalues)
    L = np.zeros((len(landmarks), dimension))
    for i in range(dimension):
        L[:, i] = eigenvectors[:, i] * np.sqrt(eigenvalues[i])

    # Append the position of the landmarks to the dataset
    landmarks = landmarks.assign(position=L.tolist(), landmark=True)

    # 3. Apply distance-based triangulation

    # The mean distance between the landmarks
    # is the mean of the columns of Deltan
    mean_distance = delta_n.mean(axis=0)

    # L_sharp is the pseudo-inverse of L
    # given by eigenvectors * 1/sqrt(eigenvalues)
    L_sharp = np.zeros((dimension, len(landmarks)))
    for i in range(dimension):
        L_sharp[i, :] = (
            eigenvectors[:, i].transpose() * 1/np.sqrt(eigenvalues[i])
        )

    # We first need to get the embeddings of the other points
    other_points = dataset[~dataset.index.isin(landmarks.index)]
    other_embeddings = np.vstack(other_points['embeddings'].apply(np.array))

    # We compute for each point the distance to the landmarks
    distance_to_landmarks = euclidean_distances(
        other_embeddings, landmark_embeddings
    )

    # Going through each point, we compute its position
    # by -1/2 * L_sharp * (distance_to_landmarks - mean_distance)
    positions = np.zeros((len(other_points), dimension))
    for i in range(len(other_points)):
        position = -1/2 * np.dot(
            L_sharp, distance_to_landmarks[i] - mean_distance
        )
        positions[i, :] = position

    # Append the position of the other points to the dataset
    other_points = other_points.assign(
        position=positions.tolist(), landmark=False
    )

    # We concatenate the landmarks and the other points
    return pd.concat([landmarks, other_points])


if __name__ == "__main__":
    main()
