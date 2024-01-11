import math
import struct
import sysv_ipc
import mmap
import pandas as pd
import io
import subprocess
from typing import Dict, Iterable, Tuple, Generator
from abc import ABC, abstractproperty


class Neighbors(ABC):
    INDEX_DISTANCE_PAIR_FORMAT: str = 'If'
    INDEX_DISTANCE_PAIR_SIZE: int = struct.calcsize(INDEX_DISTANCE_PAIR_FORMAT)

    _k: int
    _distance_metric: str
    _datapoint_amount: int
    _neighbor_view: memoryview

    def __init__(self, k: int, distance_metric: str):
        if not 0 <= k <= self._datapoint_amount:
            raise ValueError(
                f'k must be in range 0 <= k <= {self._datapoint_amount}, '
                f'but is {k}'
            )
        self._k = k
        self._distance_metric = distance_metric

    @abstractproperty
    def _available_neighbor_count(self) -> int:
        raise NotImplementedError()

    def _get_point_offset(self, point_index: int) -> int:
        return (
            point_index
            * self._available_neighbor_count
            * self.INDEX_DISTANCE_PAIR_SIZE
        )

    def get(
        self, point_index: int
    ) -> Generator[Tuple[int, float], None, None]:
        if not 0 <= point_index < self._datapoint_amount:
            raise IndexError(
                f'Point index {point_index} out of range '
                f'(0 <= index < {self._datapoint_amount})'
            )
        offset = self._get_point_offset(point_index)
        neighbors = (
            struct.unpack(
                self.INDEX_DISTANCE_PAIR_FORMAT,
                self._neighbor_view[
                    offset + (i + 1) * self.INDEX_DISTANCE_PAIR_SIZE:
                    offset + (i + 2) * self.INDEX_DISTANCE_PAIR_SIZE
                ]
            )
            for i in range(self._k)
        )
        return neighbors

    def get_many(
        self, point_indices: Iterable[int]
    ) -> Generator[Generator[Tuple[int, float], None, None], None, None]:
        return (
            self.get(point_index)
            for point_index in point_indices
        )

    def dump(self, filename: str):
        with open(filename, 'wb') as file:
            file.write(self._neighbor_view)


class Neighbors2D(Neighbors):
    SHARED_MEMORY_ACCESS_FLAGS: int = 0o666
    DISTANCE_METRICS: Dict[str, int] = {
        'euclidean': 0,
        'cosine': 1
    }

    POSITION_FORMAT: str = 'ff'
    POSITION_SIZE: int = struct.calcsize(POSITION_FORMAT)

    NEIGHBORS_EXECUTABLE: str = './neighbors2d'

    _dataset: pd.DataFrame
    _shared_memory: sysv_ipc.SharedMemory
    _neighbor_buffer: bytearray

    def __init__(self, dataset: pd.DataFrame, k: int, distance_metric: str):
        if 'position' not in dataset.columns:
            raise ValueError(
                'Dataset must contain a column "position" '
                'with 2D positions'
            )
        if distance_metric not in self.DISTANCE_METRICS:
            raise ValueError(
                'Distance metric must be one of '
                f'{self.DISTANCE_METRICS.keys()}'
            )
        self._dataset = dataset
        self._datapoint_amount = len(dataset)
        super().__init__(k, distance_metric)
        self._shared_memory = sysv_ipc.SharedMemory(
            None,
            flags=sysv_ipc.IPC_CREX,
            size=self._shared_memory_size,
            mode=self.SHARED_MEMORY_ACCESS_FLAGS
        )
        self._write_shared_memory()
        self._compute_neighbors()
        self._read_shared_memory()

    @property
    def _positions_size(self) -> int:
        return self.POSITION_SIZE * self._datapoint_amount

    @property
    def _shared_memory_size(self) -> int:
        return self._positions_size + self._index_distance_pairs_size

    @property
    def _available_neighbor_count(self) -> int:
        return min(self._k + 1, self._datapoint_amount)

    @property
    def _index_distance_pairs_size(self) -> int:
        return (
            self.INDEX_DISTANCE_PAIR_SIZE
            * self._datapoint_amount
            * self._available_neighbor_count
        )

    def _write_shared_memory(self):
        buffer = bytearray(self._shared_memory_size)
        for index, row in self._dataset.iterrows():
            offset = index * self.POSITION_SIZE
            struct.pack_into(
                self.POSITION_FORMAT, buffer, offset,
                row['position'][0], row['position'][1]
            )
        self._shared_memory.write(buffer)

    def _compute_neighbors(self):
        process = subprocess.Popen(
            [
                self.NEIGHBORS_EXECUTABLE,
                str(self._shared_memory.key),
                str(self._shared_memory.size),
                str(self.DISTANCE_METRICS[self._distance_metric]),
                str(self._datapoint_amount),
                str(self._k)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.wait()
        if process.returncode != 0:
            _, stderr = process.communicate()
            raise RuntimeError(
                f'neighbors2d returned with code {process.returncode}!\n'
                f'stderr:\n{stderr}'
            )

    def _read_shared_memory(self):
        self._neighbor_buffer = bytearray(self._shared_memory.read(
            self._index_distance_pairs_size,
            self._positions_size
        ))
        self._shared_memory.remove()
        self._neighbor_view = memoryview(self._neighbor_buffer)


class NeighborsHighD(Neighbors):

    FROM_START_OF_FILE: int = 0
    FROM_END_OF_FILE: int = 2
    ENTIRE_FILE: int = 0

    INDEX_DISTANCE_PAIR_FORMAT: str = 'If'
    INDEX_DISTANCE_PAIR_SIZE: int = struct.calcsize(INDEX_DISTANCE_PAIR_FORMAT)

    FILENAME: str = '/server/data/imdb_{distance_metric}_neighbors.bin'

    _file: io.BufferedReader
    _memory_map: mmap.mmap

    def __init__(self, k: int, distance_metric: str):
        self._memory_map = None
        filename = self.FILENAME.format(distance_metric=distance_metric)
        self._file = open(filename, 'rb')
        self._get_datapoint_amount()
        super().__init__(k, distance_metric)
        self._map_file()

    def __del__(self):
        self._neighbor_view.release()
        if self._memory_map is not None:
            self._memory_map.close()
        self._file.close()

    @property
    def _available_neighbor_count(self) -> int:
        return self._datapoint_amount

    def _get_datapoint_amount(self) -> int:
        self._file.seek(0, self.FROM_END_OF_FILE)
        file_size = self._file.tell()
        self._file.seek(0, self.FROM_START_OF_FILE)
        self._datapoint_amount = (
            int(math.sqrt(file_size / self.INDEX_DISTANCE_PAIR_SIZE))
        )

    def _map_file(self):
        self._memory_map = mmap.mmap(
            self._file.fileno(),
            self.ENTIRE_FILE,
            access=mmap.ACCESS_READ
        )
        self._neighbor_view = memoryview(self._memory_map)


if __name__ == '__main__':
    # Set the filenames to work outside of Docker:
    Neighbors2D.NEIGHBORS_EXECUTABLE = (
        './services/backend/neighbors2d'
    )
    NeighborsHighD.FILENAME = (
        './volumes/data/imdb_{distance_metric}_neighbors.bin'
    )

    # Examples:
    print("Get the 10 nearest cosine high D neighbors of point 0:")
    neighbors = NeighborsHighD(10, 'cosine')
    print("\n".join(str(n) for n in neighbors.get(0)))

    print()

    print("Get the 7 nearest euclidean high D neighbors of point 42:")
    neighbors = NeighborsHighD(7, 'euclidean')
    print("\n".join(str(n) for n in neighbors.get(42)))

    print()

    print("Get the 5 nearest euclidean 2D neighbors of point 1:")
    dataset = pd.DataFrame([
        {'position': (0, 0)},
        {'position': (1, 0)},
        {'position': (0, 1)},
        {'position': (1, 1)},
        {'position': (2, 2)},
        {'position': (3, 3)},
        {'position': (4, 4)},
    ])
    neighbors = Neighbors2D(dataset, 5, 'euclidean')
    print("\n".join(str(n) for n in neighbors.get(1)))

    print()

    print("Get the 7 nearest euclidean 2D cosine of all points:")
    from random import random
    dataset = pd.DataFrame([
        {'position': (random(), random())}
        for _ in range(25000)
    ])
    # This line takes about 37 seconds on my machine:
    neighbors = Neighbors2D(dataset, 7, 'cosine')
    for neighbors_generator in neighbors.get_many(range(len(dataset))):
        print(", ".join(str(n) for n in neighbors_generator))
