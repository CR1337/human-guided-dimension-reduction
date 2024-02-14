#ifndef __EUCLIDEAN_HPP__
#define __EUCLIDEAN_HPP__

#include <stdlib.h>

#include "types.hpp"

void findEuclideanNeighbors2D(
    Position2D *positions,
    size_t datapointAmount,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
);
void findEuclideanNeighbors768D(
    Position768D *positions,
    size_t datapointAmount,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
);

#endif // __EUCLIDEAN_HPP__
