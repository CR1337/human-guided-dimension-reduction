import io
import gc
import os
import mmap
import struct
import sysv_ipc
import subprocess
import pandas as pd
from abc import ABC
from itertools import islice
from typing import Dict, Generator, Tuple, List


DistanceIndexPairGenerator = Generator[Tuple[int, float], None, None]
RanksGenerator = Generator[List[int], None, None]

inside_docker: bool = bool(os.environ.get('INSIDE_DOCKER', False))


class Neighbors(ABC):

    DIMENSIONS_2D: int = 2
    DIMENSIONS_768: int = 768

    PARAMETER_FORMAT: str = "=bHH"
    POSITION_2D_FORMAT: str = f"={DIMENSIONS_2D}f"
    POSITION_768D_FORMAT: str = f"={DIMENSIONS_768}f"
    DISTANCE_INDEX_PAIR_FORMAT: str = "=Hf"
    INDEX_FORMAT: str = "=H"

    PARAMETER_SIZE: int = struct.calcsize(PARAMETER_FORMAT)
    POSITION_2D_SIZE: int = struct.calcsize(POSITION_2D_FORMAT)
    POSITION_768D_SIZE: int = struct.calcsize(POSITION_768D_FORMAT)
    DISTANCE_INDEX_PAIR_SIZE: int = struct.calcsize(DISTANCE_INDEX_PAIR_FORMAT)
    INDEX_SIZE: int = struct.calcsize(INDEX_FORMAT)

    DISTANCE_METRICS: Dict[str, str] = {
        "euclidean": "e",
        "cosine": "c"
    }
    REVERSE_DISTANCE_METRICS: Dict[str, str] = {
        "e": "euclidean",
        "c": "cosine"
    }

    _distance_metric: str
    _datapoint_amount: int
    _dimensions: int

    _memory_view: memoryview

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @property
    def datapoint_amount(self) -> int:
        return self._datapoint_amount

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def __init__(
        self,
        distance_metric: str,
        datapoint_amount: int,
        dimensions: int
    ):
        self._memory_view = None
        self._raise_for_distance_metric(distance_metric)
        self._distance_metric = distance_metric
        self._raise_for_datapoint_amount(datapoint_amount)
        self._datapoint_amount = datapoint_amount
        self._raise_for_dimensions(dimensions)
        self._dimensions = dimensions

    def __del__(self):
        if self._memory_view is not None:
            self._memory_view.release()

    def _raise_for_distance_metric(self, distance_metric: str):
        if distance_metric not in self.DISTANCE_METRICS:
            raise ValueError(
                f"Invalid distance metric: {distance_metric}. "
                f"Valid distance metrics: {self.DISTANCE_METRICS.keys()}"
            )

    def _raise_for_datapoint_amount(self, datapoint_amount: int):
        if datapoint_amount <= 0:
            raise ValueError(
                f"Invalid datapoint amount: {datapoint_amount}. "
                "Datapoint amount must be > 0."
            )

    def _raise_for_dimensions(self, dimensions: int):
        if dimensions not in (self.DIMENSIONS_2D, self.DIMENSIONS_768):
            raise ValueError(
                f"Invalid dimensions: {dimensions}. "
                f"Valid dimensions: {self.DIMENSIONS_2D},"
                f" {self.DIMENSIONS_768}"
            )

    @property
    def _position_size(self) -> int:
        if self._dimensions == self.DIMENSIONS_2D:
            return self.POSITION_2D_SIZE
        elif self._dimensions == self.DIMENSIONS_768:
            return self.POSITION_768D_SIZE

    @property
    def _positions_size(self) -> int:
        return self._position_size * self._datapoint_amount

    @property
    def _distance_index_pairs_size(self) -> int:
        return (
            self.DISTANCE_INDEX_PAIR_SIZE
            * self._datapoint_amount ** 2
        )

    @property
    def _ranks_size(self) -> int:
        return self.INDEX_SIZE * self._datapoint_amount ** 2

    @property
    def _positions_offset(self) -> int:
        return self.PARAMETER_SIZE

    def _get_distance_index_pairs_offset(self, index: int) -> int:
        return (
            self._positions_offset + self._positions_size
            + self.DISTANCE_INDEX_PAIR_SIZE * self._datapoint_amount * index
        )

    def _get_ranks_offset(self, index: int) -> int:
        return (
            self._positions_offset + self._positions_size
            + self._distance_index_pairs_size
            + self.INDEX_SIZE * self._datapoint_amount * index
        )

    def _raise_for_index(self, index: int) -> None:
        if index >= self._datapoint_amount or index < 0:
            raise IndexError(
                f"Index {index} out of range (0, {self._datapoint_amount})"
            )

    def get_position(self, index: int) -> Tuple[float, ...]:
        """
        Returns the position of the datapoint at the given index.

        :param index: The index of the datapoint.

        :return: The position of the datapoint as tuple of floats.
        """
        self._raise_for_index(index)
        offset = self._positions_offset + self._position_size * index
        position_format = (
            self.POSITION_2D_FORMAT
            if self._dimensions == self.DIMENSIONS_2D
            else self.POSITION_768D_FORMAT
        )
        position = struct.unpack(
            position_format,
            self._memory_view[
                offset: offset + self._position_size
            ]
        )
        return position

    def get_neighbors(self, index: int) -> DistanceIndexPairGenerator:
        """
        Returns the nearest neighbors of the datapoint at the given index.

        :param index: The index of the datapoint.

        :return: The nearest neighbors of the datapoint as tuple of
            (index, distance) pairs.
        """
        self._raise_for_index(index)
        offset = self._get_distance_index_pairs_offset(index)
        neighbors = (
            struct.unpack(
                self.DISTANCE_INDEX_PAIR_FORMAT,
                self._memory_view[
                    offset + self.DISTANCE_INDEX_PAIR_SIZE * (i + 1):
                    offset + self.DISTANCE_INDEX_PAIR_SIZE * (i + 2)
                ]
            )
            for i in range(self._datapoint_amount)
        )
        return neighbors

    def get_k_neighbors(
        self, index: int, k: int
    ) -> DistanceIndexPairGenerator:
        return islice(self.get_neighbors(index), k)

    def _get_ranks(self, index: int) -> List[int]:
        offset = self._get_ranks_offset(index)
        ranks = [
            struct.unpack(
                self.INDEX_FORMAT,
                self._memory_view[
                    offset + self.INDEX_SIZE * i:
                    offset + self.INDEX_SIZE * (i + 1)
                ]
            )[0]
            for i in range(self._datapoint_amount)
        ]
        return ranks

    def get_ranks(self) -> RanksGenerator:
        """
        Returns the ranks of all datapoints.
        The rank a a points nearest neighbor is 1. The rank of `0`
        refers to the point itself.

        :return: The ranks of all datapoints as generator of lists of integers.
        """
        return (
            self._get_ranks(i)
            for i in range(self._datapoint_amount)
        )


class ComputedNeighbors(Neighbors):

    SHARED_MEMORY_ACCESS_MODE: int = 0o666

    NEIGHBORS_EXECUTABLE_PATH: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "neighbors", "neighbors"
    )

    _dataset: pd.DataFrame

    _shared_memory: sysv_ipc.SharedMemory

    def __init__(
        self,
        distance_metric: str,
        dimensions: int,
        dataset: pd.DataFrame
    ):
        self._raise_for_dataset(dataset, dimensions)
        self._dataset = dataset

        super().__init__(
            distance_metric,
            len(dataset),
            dimensions
        )

        self._shared_memory = sysv_ipc.SharedMemory(
            None,
            flags=sysv_ipc.IPC_CREX,
            size=self._shared_memory_size,
            mode=self.SHARED_MEMORY_ACCESS_MODE
        )

        self._write_shared_memory()
        self._compute_neighbors()
        self._read_shared_memory()

    def __del__(self):
        super().__del__()
        if self._shared_memory is not None:
            self._shared_memory.remove()
        gc.collect()

    @property
    def _input_size(self) -> int:
        return self.PARAMETER_SIZE + self._positions_size

    @property
    def _output_size(self) -> int:
        return self._distance_index_pairs_size + self._ranks_size

    @property
    def _shared_memory_size(self) -> int:
        return self._input_size + self._output_size

    def _raise_for_dataset(self, dataset: pd.DataFrame, dimensions: int):
        if dimensions == self.DIMENSIONS_2D:
            if 'position' not in dataset.columns:
                raise ValueError(
                    "Invalid dataset: position column missing."
                )
            if len(dataset['position'][0]) != self.DIMENSIONS_2D:
                raise ValueError(
                    "Invalid dataset: positions must have "
                    f"{self.DIMENSIONS_2D} dimensions."
                )
        elif dimensions == self.DIMENSIONS_768:
            if 'embeddings' not in dataset.columns:
                raise ValueError(
                    "Invalid dataset: embedding column missing."
                )
            if len(dataset['embeddings'][0]) != self.DIMENSIONS_768:
                raise ValueError(
                    "Invalid dataset: embeddings must have "
                    f"{self.DIMENSIONS_768} dimensions."
                )

    def _write_shared_memory(self):
        buffer = bytearray(self._input_size)
        struct.pack_into(
            self.PARAMETER_FORMAT,
            buffer,
            0,
            ord(self.DISTANCE_METRICS[self._distance_metric]),
            self._datapoint_amount,
            self._dimensions
        )
        position_key = (
            'position'
            if self._dimensions == self.DIMENSIONS_2D
            else 'embeddings'
        )
        position_format = (
            self.POSITION_2D_FORMAT
            if self._dimensions == self.DIMENSIONS_2D
            else self.POSITION_768D_FORMAT
        )
        for i, row in self._dataset.iterrows():
            offset = self._positions_offset + self._position_size * i
            struct.pack_into(
                position_format,
                buffer,
                offset,
                *row[position_key]
            )
        self._shared_memory.write(buffer)

    def _compute_neighbors(self):
        process = subprocess.Popen(
            [
                self.NEIGHBORS_EXECUTABLE_PATH,
                str(self._shared_memory.key),
                str(self._shared_memory.size)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.wait()
        stdout, stderr = process.communicate()
        print(stdout.decode("utf-8"))
        if process.returncode != 0:
            raise RuntimeError(
                f'{self.NEIGHBORS_EXECUTABLE_PATH} returned with '
                f'code {process.returncode}!\n'
                f'stderr:\n{stderr.decode("utf-8")}'
            )

    def _read_shared_memory(self):
        buffer = bytearray(self._shared_memory.read(self._shared_memory_size))
        self._shared_memory.remove()
        self._shared_memory = None
        self._memory_view = memoryview(buffer)

    def dump(self, filename: str):
        with open(filename, 'wb') as file:
            file.write(self._memory_view)


class CachedNeighbors(Neighbors):

    ENTIRE_FILE: int = 0
    SMALL_SUFFIX: str = "_small"
    ALL_NEIGHBORS_768D_FILENAME: str = (
        "/server/data/imdb_{distance_metric}_neighbors{small_suffix}.bin"
        if inside_docker
        else "volumes/data/imdb_{distance_metric}_neighbors{small_suffix}.bin"
    )

    _file: io.BufferedReader
    _memory_map: mmap.mmap

    @classmethod
    def all_neighbors_768d(cls, distance_metric: str, use_small: bool = False):
        return cls(cls.ALL_NEIGHBORS_768D_FILENAME.format(
            distance_metric=distance_metric,
            small_suffix=cls.SMALL_SUFFIX if use_small else ""
        ))

    def __init__(self, filename: str):
        self._memory_map = None
        self._memory_view = None
        self._file = None

        self._file = open(filename, 'rb')
        super().__init__(*self._read_parameters())
        self._map_file()

    def __del__(self):
        super().__del__()
        if self._memory_map is not None:
            self._memory_map.close()
        if self._file is not None:
            self._file.close()
        gc.collect()

    def _read_parameters(self) -> Tuple[str, int, int, int]:
        buffer = self._file.read(self.PARAMETER_SIZE)
        paremeters = struct.unpack(self.PARAMETER_FORMAT, buffer)
        parameters = (
            self.REVERSE_DISTANCE_METRICS[chr(paremeters[0])],
            *paremeters[1:]
        )
        return parameters

    def _map_file(self):
        self._memory_map = mmap.mmap(
            self._file.fileno(),
            self.ENTIRE_FILE,
            access=mmap.ACCESS_READ
        )
        self._memory_view = memoryview(self._memory_map)


if __name__ == "__main__":
    # Change directory for testing outside of docker
    CachedNeighbors.ALL_NEIGHBORS_768D_FILENAME = (
        "./volumes/data/imdb_{distance_metric}_neighbors{small_suffix}.bin"
    )

    # dummy dataset
    dataset_2d = pd.DataFrame([
        {'position': (0.1, 0.9)},
        {'position': (1.0, 0.0)},
        {'position': (0.0, 1.0)},
        {'position': (1.0, 1.0)},
        {'position': (2.0, 2.0)},
        {'position': (3.0, 3.0)},
        {'position': (4.0, 4.0)},
    ])

    # compute nearest euclidean neighbors and all ranks
    euclidean_neighbors_2d = ComputedNeighbors(
        distance_metric="euclidean",
        dimensions=Neighbors.DIMENSIONS_2D,
        dataset=dataset_2d
    )
    euclidean_neighbors_2d.dump("euclidean.bin")
    for point_index in range(len(dataset_2d)):
        print(euclidean_neighbors_2d.get_position(point_index))
        print(list(euclidean_neighbors_2d.get_neighbors(point_index)))
        print(list(euclidean_neighbors_2d.get_ranks()))
        print()
    # if you don't need it anymore, you should delete it
    del euclidean_neighbors_2d

    print()
    print()
    print()

    # compute nearest cosine neighbors and all ranks
    cosine_neighbors_2d = ComputedNeighbors(
        distance_metric="cosine",
        dimensions=Neighbors.DIMENSIONS_2D,
        dataset=dataset_2d
    )
    cosine_neighbors_2d.dump("cosine.bin")
    for point_index in range(len(dataset_2d)):
        print(cosine_neighbors_2d.get_position(point_index))
        print(list(cosine_neighbors_2d.get_neighbors(point_index)))
        print(list(cosine_neighbors_2d.get_ranks()))
        print()
    # if you don't need it anymore, you should delete it
    del cosine_neighbors_2d

    print()
    print()
    print()

    # exit()

    # load cached euclidean neighbors
    euclidean_neighbors_768d = CachedNeighbors.all_neighbors_768d(
        distance_metric="euclidean", use_small=True
    )
    # You can access it like the ComputedNeighbors class
    ranks = euclidean_neighbors_768d.get_ranks()
    print(next(ranks))
    print(next(ranks))

    # if you don't need it anymore, you should delete it
    del euclidean_neighbors_768d

    print()
    print()
    print()

    # load cached cosine neighbors
    cosine_neighbors_768d = CachedNeighbors.all_neighbors_768d(
        distance_metric="cosine", use_small=True
    )
    # You can access it like the ComputedNeighbors class
    ranks = cosine_neighbors_768d.get_ranks()
    print(next(ranks))
    print(next(ranks))
    print(next(ranks))

    # if you don't need it anymore, you should delete it
    del cosine_neighbors_768d
