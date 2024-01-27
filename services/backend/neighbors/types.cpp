#include "types.hpp"

int compareDistanceIndexPair(const void *a, const void *b) {
    const DistanceIndexPair *distanceIndexPairA = (const DistanceIndexPair*)a;
    const DistanceIndexPair *distanceIndexPairB = (const DistanceIndexPair*)b;
    const float difference = distanceIndexPairA->distance - distanceIndexPairB->distance;
    return (difference > 0.0f) - (difference < 0.0f);
}