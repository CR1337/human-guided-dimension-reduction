import math
import struct
from typing import List, Tuple


class Neighbors:

    FROM_START_OF_FILE: int = 0
    FROM_END_OF_FILE: int = 2

    STRUCT_FORMAT: str = 'If'
    STRUCT_SIZE: int = struct.calcsize(STRUCT_FORMAT)

    FILENAME: str = '/server/data/imdb_{distance_metric}_neighbors.bin'

    @classmethod
    def _get_point_amount(cls, file) -> int:
        file.seek(0, cls.FROM_END_OF_FILE)
        file_size = file.tell()
        file.seek(0, cls.FROM_START_OF_FILE)
        return int(math.sqrt(file_size / cls.STRUCT_SIZE))

    @classmethod
    def _get_point_offset(cls, point_index: int, point_amount: int) -> int:
        return point_index * cls.STRUCT_SIZE * point_amount

    @classmethod
    def _raise_on_point_index_and_k(
        cls, point_index: int, k: int, point_amount: int
    ):
        if point_index < 0 or point_index >= point_amount:
            raise ValueError(
                f'`point_index` must be between 0 and {point_amount - 1}!'
            )
        if k < 0 or k >= point_amount:
            raise ValueError(
                f'`k` must be between 0 and {point_amount - 1}!'
            )

    @classmethod
    def get(
        cls, point_index: int, k: int, distance_metric: str
    ) -> List[Tuple[int, float]]:
        '''
        Returns the `k` nearest neighbors of the point with index `point_index`
        using the distance metric `distance_metric`.

        The neighbors are returned as a list of tuples, where the first element
        is the index of the neighbor and the second element is the distance to
        the neighbor.

        The neighbors are sorted by distance, with the nearest neighbor first.

        Raises a `ValueError` if `point_index` or `k` are out of bounds.
        '''
        filename = cls.FILENAME.format(distance_metric=distance_metric)

        with open(filename, 'rb') as file:
            point_amount = cls._get_point_amount(file)
            cls._raise_on_point_index_and_k(point_index, k, point_amount)
            offset = cls._get_point_offset(point_index, point_amount)

            file.seek(offset + cls.STRUCT_SIZE, cls.FROM_START_OF_FILE)
            neighbors = [
                struct.unpack(cls.STRUCT_FORMAT, file.read(cls.STRUCT_SIZE))
                for _ in range(k)
            ]

        return neighbors


if __name__ == '__main__':
    # Set the filename to work outside of docker
    Neighbors.FILENAME = './volumes/data/imdb_{distance_metric}_neighbors.bin'

    # Examples:

    print("Get the 10 nearest cosine neighbors of point 0:")
    print("\n".join(str(n) for n in Neighbors.get(0, 10, 'cosine')))

    print()

    print("Get the 7 nearest euclidean neighbors of point 42")
    print("\n".join(str(n) for n in Neighbors.get(42, 7, 'euclidean')))
