#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <cstdint>

#define DIMENSIONS_2 (2)
#define DIMENSIONS_768 (768)

#define EUCLIDEAN_DISTANCE_METRIC ('e')
#define COSINE_DISTANCE_METRIC ('c')

typedef int8_t DistanceMetric;
typedef uint16_t Index;
typedef uint16_t DimensionCount;

typedef struct __attribute__((packed)) {
    DistanceMetric distanceMetric;
    Index datapointAmount;
    Index k;
    DimensionCount dimensions;
} Parameters;

typedef struct __attribute__((packed)) {
    float x;
    float y;
} Position2D;

typedef struct __attribute__((packed)) {
    float values[DIMENSIONS_768];
} Position768D;

typedef struct __attribute__((packed)) {
    Index index;
    float distance;
} DistanceIndexPair;

static int compareDistanceIndexPair(const void *a, const void *b) {
    const DistanceIndexPair *distanceIndexPairA = (const DistanceIndexPair*)a;
    const DistanceIndexPair *distanceIndexPairB = (const DistanceIndexPair*)b;
    const float difference = distanceIndexPairA->distance - distanceIndexPairB->distance;
    return (difference > 0.0f) - (difference < 0.0f);
}

#endif // __TYPES_HPP__
