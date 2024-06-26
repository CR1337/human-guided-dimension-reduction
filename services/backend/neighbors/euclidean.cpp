#include "euclidean.hpp"
#include "util.hpp"

#include <sys/sysinfo.h>
#include <pthread.h>
#include <math.h>

typedef struct {
    size_t offset;
    Position2D *positions;
    DistanceIndexPair *distanceIndexPairs;
    Index *ranks;
    size_t coreAmount;
    size_t datapointAmount;
} EuclideanThreadArgs2D;

typedef struct {
    size_t offset;
    Position768D *positions;
    DistanceIndexPair *distanceIndexPairs;
    Index *ranks;
    size_t coreAmount;
    size_t datapointAmount;
} EuclideanThreadArgs768D;

float euclideanDistance2D(const Position2D *a, const Position2D *b) {
    Position2D difference = (Position2D){
        .x = a->x - b->x,
        .y = a->y - b->y
    };
    return hypotf(difference.x, difference.y);
}

float euclideanDistance768D(const Position768D *a, const Position768D *b) {
    float sum = 0.0f;
    for (size_t i = 0; i < DIMENSIONS_768; ++i) {
        float difference = a->values[i] - b->values[i];
        sum += difference * difference;
    }
    return sqrtf(sum);
}

void * euclideanThreadHandler2D(void *args) {
    const EuclideanThreadArgs2D *threadArgs = (EuclideanThreadArgs2D*)args;
    const size_t start = threadArgs->offset;
    const Position2D *positions = threadArgs->positions;
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
                .distance = euclideanDistance2D(positions + i, positions + j)
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

void * euclideanThreadHandler768D(void *args) {
    const EuclideanThreadArgs768D *threadArgs = (EuclideanThreadArgs768D*)args;
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
                .distance = euclideanDistance768D(positions + i, positions + j)
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

void findEuclideanNeighbors2D(
    Position2D *positions,
    size_t datapointAmount,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
) {
    size_t coreAmount = get_nprocs();
    if (coreAmount > datapointAmount) coreAmount = datapointAmount;
    pthread_t threads[coreAmount];
    EuclideanThreadArgs2D threadArgs[coreAmount];

    for (size_t i = 0; i < coreAmount; ++i) {
        threadArgs[i] = (EuclideanThreadArgs2D) {
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
        pthread_create(&threads[i], NULL, euclideanThreadHandler2D, &threadArgs[i]);
    }

    for (size_t i = 0; i < coreAmount; ++i) {
        pthread_join(threads[i], NULL);
    }
}

void findEuclideanNeighbors768D(
    Position768D *positions,
    size_t datapointAmount,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
) {
    size_t coreAmount = get_nprocs();
    if (coreAmount > datapointAmount) coreAmount = datapointAmount;
    pthread_t threads[coreAmount];
    EuclideanThreadArgs768D threadArgs[coreAmount];

    for (size_t i = 0; i < coreAmount; ++i) {
        threadArgs[i] = (EuclideanThreadArgs768D) {
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
        pthread_create(&threads[i], NULL, euclideanThreadHandler768D, &threadArgs[i]);
    }

    for (size_t i = 0; i < coreAmount; ++i) {
        pthread_join(threads[i], NULL);
    }
}
