#ifndef __COSINE_HPP__
#define __COSINE_HPP__

#include <stdlib.h>

#include "types.hpp"

void findCosineNeighbors2D(
    Position2D *positions,
    size_t datapointAmount,
    size_t k,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
);
void findCosineNeighbors768D(
    Position768D *positions,
    size_t datapointAmount,
    size_t k,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
);

#endif // __COSINE_HPP__
