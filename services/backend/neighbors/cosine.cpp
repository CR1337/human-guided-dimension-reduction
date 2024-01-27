#include "cosine.hpp"
#include "util.hpp"

#include <sys/sysinfo.h>
#include <pthread.h>

#include <algorithm>
#include <cmath>
#include <vector>

typedef struct {
    size_t offset;
    DistanceIndexPair *distanceIndexPairs;
    Index *ranks;
    size_t coreAmount;
    size_t datapointAmount;
    std::vector<std::pair<Position2D*, float>> *positionAngles;
} CosineThreadArgs2D;

typedef struct {
    size_t offset;
    Position768D *positions;
    DistanceIndexPair *distanceIndexPairs;
    Index *ranks;
    size_t coreAmount;
    size_t datapointAmount;
} CosineThreadArgs768D;

float positionAngle2D(const Position2D *a) {
    return atan2f(a->y, a->x);
}

float cosineDistance2D(const Position2D *a, const Position2D *b) {
    float dotAA = a->x * a->x + a->y * a->y;
    float dotBB = b->x * b->x + b->y * b->y;
    float dotAB = a->x * b->x + a->y * b->y;
    return 1.0f - dotAB / sqrtf(dotAA * dotBB);
}

float cosineDistance768D(const Position768D *a, const Position768D *b) {
    float dotAA = 0.0f;
    float dotBB = 0.0f;
    float dotAB = 0.0f;
    for (size_t i = 0; i < DIMENSIONS_768; ++i) {
        dotAA += a->values[i] * a->values[i];
        dotBB += b->values[i] * b->values[i];
        dotAB += a->values[i] * b->values[i];
    }
    return 1.0f - dotAB / sqrtf(dotAA * dotBB);
}

float relativeAngle(const float a, const float b) {
    float diff = std::fabs(a - b);
    if (diff > M_PI) diff = 2 * M_PI - diff;
    return std::fabs(diff);
}

void * cosineThreadHandler2D(void *args) {
    const CosineThreadArgs2D *threadArgs = (CosineThreadArgs2D*)args;
    const size_t start = threadArgs->offset;
    DistanceIndexPair *distanceIndexPairs = threadArgs->distanceIndexPairs;
    Index *ranks = threadArgs->ranks;
    const size_t coreAmount = threadArgs->coreAmount;
    const size_t datapointAmount = threadArgs->datapointAmount;
    const std::vector<std::pair<Position2D*, float>> *positionAngles = threadArgs->positionAngles;

    size_t end = (
        start
        + (datapointAmount / coreAmount)
        + (datapointAmount % coreAmount != 0)
    );
    if (end > datapointAmount) end = datapointAmount;

    for (size_t i = start; i < end; ++i) {
        auto [position, angle] = (*positionAngles)[i];
        Index leftIndex = i;
        Index rightIndex = (i == datapointAmount - 1) ? 0 : i + 1;
        for (size_t j = 0; j < datapointAmount; ++j) {
            const float leftAngle = (*positionAngles)[leftIndex].second;
            const float rightAngle = (*positionAngles)[rightIndex].second;
            if (relativeAngle(leftAngle, angle) < relativeAngle(rightAngle, angle)) {
                float distance = cosineDistance2D(position, (*positionAngles)[leftIndex].first);
                distanceIndexPairs[i * datapointAmount + j] = (DistanceIndexPair){
                    .index = leftIndex,
                    .distance = distance
                };
                leftIndex = (leftIndex == 0) ? datapointAmount - 1 : leftIndex - 1;
            } else {
                float distance = cosineDistance2D(position, (*positionAngles)[rightIndex].first);
                distanceIndexPairs[i * datapointAmount + j] = (DistanceIndexPair){
                    .index = rightIndex,
                    .distance = distance
                };
                rightIndex = (rightIndex == datapointAmount - 1) ? 0 : rightIndex + 1;
            }
        }
        for (size_t j = 0; j < datapointAmount; ++j) {
            ranks[i * datapointAmount + distanceIndexPairs[i * datapointAmount + j].index] = j;
        }
    }

    return nullptr;
}

void * cosineThreadHandler768D(void *args) {
    const CosineThreadArgs768D *threadArgs = (CosineThreadArgs768D*)args;
    const size_t start = threadArgs->offset;
    const Position768D *positions = threadArgs->positions;
    DistanceIndexPair *distanceIndexPairs = threadArgs->distanceIndexPairs;
    Index *ranks = threadArgs->ranks;
    const size_t coreAmount = threadArgs->coreAmount;
    const size_t datapointAmount = threadArgs->datapointAmount;

    size_t end = (
        start
        + (datapointAmount / coreAmount)
        + (datapointAmount % coreAmount != 0)
    );
    if (end > datapointAmount) end = datapointAmount;

    for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < datapointAmount; ++j) {
            distanceIndexPairs[i * datapointAmount + j] = (DistanceIndexPair){
                .index = (Index)j,
                .distance = cosineDistance768D(positions + i, positions + j)
            };
        }
        qsort(
            distanceIndexPairs + i * datapointAmount,
            datapointAmount,
            sizeof(DistanceIndexPair),
            compareDistanceIndexPair
        );
        for (size_t j = 0; j < datapointAmount; ++j) {
            ranks[i * datapointAmount + distanceIndexPairs[i * datapointAmount + j].index] = j;
        }
    }

    return nullptr;
}

void findCosineNeighbors2D(
    Position2D *positions,
    size_t datapointAmount,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
) {
    std::vector<std::pair<Position2D*, float>> positionAngles;
    for (size_t i = 0; i < datapointAmount; ++i) {
        Position2D *position = &positions[i];
        float angle = positionAngle2D(position);
        positionAngles.push_back(std::make_pair(position, angle));
    }
    std::stable_sort(positionAngles.begin(), positionAngles.end(), [](const std::pair<Position2D*, float> &a, const std::pair<Position2D*, float> &b) {
        return a.second < b.second;
    });

    size_t coreAmount = get_nprocs();
    if (coreAmount > datapointAmount) coreAmount = datapointAmount;
    pthread_t threads[coreAmount];
    CosineThreadArgs2D threadArgs[coreAmount];

    for (size_t i = 0; i < coreAmount; ++i) {
        threadArgs[i] = (CosineThreadArgs2D) {
            .offset = i * (
                (datapointAmount / coreAmount)
                + (datapointAmount % coreAmount != 0)
            ),
            .distanceIndexPairs = distanceIndexPairs,
            .ranks = ranks,
            .coreAmount = coreAmount,
            .datapointAmount = datapointAmount,
            .positionAngles = &positionAngles
        };
        pthread_create(&threads[i], NULL, cosineThreadHandler2D, &threadArgs[i]);
    }

    for (size_t i = 0; i < coreAmount; ++i) {
        pthread_join(threads[i], NULL);
    }
}

void findCosineNeighbors768D(
    Position768D *positions,
    size_t datapointAmount,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
) {
    size_t coreAmount = get_nprocs();
    if (coreAmount > datapointAmount) coreAmount = datapointAmount;
    pthread_t threads[coreAmount];
    CosineThreadArgs768D threadArgs[coreAmount];

    for (size_t i = 0; i < coreAmount; ++i) {
        threadArgs[i] = (CosineThreadArgs768D) {
            .offset = i * (
                (datapointAmount / coreAmount)
                + (datapointAmount % coreAmount != 0)
            ),
            .positions = positions,
            .distanceIndexPairs = distanceIndexPairs,
            .ranks = ranks,
            .coreAmount = coreAmount,
            .datapointAmount = datapointAmount
        };
        pthread_create(&threads[i], NULL, cosineThreadHandler768D, &threadArgs[i]);
    }

    for (size_t i = 0; i < coreAmount; ++i) {
        pthread_join(threads[i], NULL);
    }
}
