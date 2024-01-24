#include "euclidean.hpp"
#include "quadtree.hpp"
#include "util.hpp"

#include <sys/sysinfo.h>
#include <pthread.h>

typedef struct {
    size_t offset;
    Position2D *positions;
    DistanceIndexPair *distanceIndexPairs;
    Index *ranks;
    size_t coreAmount;
    size_t datapointAmount;
    size_t k;
    Quadtree *quadtree;
} EuclideanThreadArgs2D;

typedef struct {
    size_t offset;
    Position768D *positions;
    DistanceIndexPair *distanceIndexPairs;
    Index *ranks;
    size_t coreAmount;
    size_t datapointAmount;
    size_t k;
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
    const size_t k = threadArgs->k;
    const Quadtree *quadtree = threadArgs->quadtree;

    size_t end = (
        start
        + (datapointAmount / coreAmount)
        + (datapointAmount % coreAmount != 0)
    );
    if (end > datapointAmount) end = datapointAmount;

    for (size_t i = start; i < end; ++i) {
        const Position2D *position = &positions[i];
        std::vector<Index> values;
        quadtree->findNearestNeighbors(position->x, position->y, datapointAmount, &values);
        for (size_t j = 0; j < k + 1; ++j) {
            distanceIndexPairs[i * (k + 1) + j] = (DistanceIndexPair){
                .index = values[j],
                .distance = euclideanDistance2D(position, &positions[values[j]])
            };
            ranks[i * datapointAmount + values[j]] = j;
        }
        for (size_t j = k + 1; j < datapointAmount; ++j) {
            ranks[i * datapointAmount + values[j]] = j;
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
    const size_t k = threadArgs->k;

    size_t end = (
        start
        + (datapointAmount / coreAmount)
        + (datapointAmount % coreAmount != 0)
    );
    if (end > datapointAmount) end = datapointAmount;

    for (size_t i = start; i < end; ++i) {
        for (size_t j = 0; j < k + 1; ++j) {
            distanceIndexPairs[i * (k + 1) + j] = (DistanceIndexPair){
                .index = j,
                .distance = euclideanDistance768D(positions + i, positions + j)
            };
            ranks[i * datapointAmount + j] = j;
        }
        for (size_t j = k + 1; j < datapointAmount; ++j) {
            ranks[i * datapointAmount + j] = j;
        }
        qsort(
            distanceIndexPairs,
            k + 1,
            sizeof(DistanceIndexPair),
            compareDistanceIndexPair
        );
    }

    return nullptr;
}

void findEuclideanNeighbors2D(
    Position2D *positions,
    size_t datapointAmount,
    size_t k,
    DistanceIndexPair *distanceIndexPairs,
    Index *ranks
) {
    float minX = 99999999;
    float minY = 99999999;
    float maxX = -99999999;
    float maxY = -99999999;
    for (size_t i = 0; i < datapointAmount; ++i) {
        Position2D *position = &positions[i];
        if (position->x < minX) minX = position->x;
        if (position->x > maxX) maxX = position->x;
        if (position->y < minY) minY = position->y;
        if (position->y > maxY) maxY = position->y;
    }
    Quadtree *quadtree = new Quadtree(minX, minY, maxX, maxY);
    for (size_t i = 0; i < datapointAmount; ++i) {
        Position2D *position = &positions[i];
        quadtree->insert(position->x, position->y, i);
    }

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
            .datapointAmount = datapointAmount,
            .k = k,
            .quadtree = quadtree
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
    size_t k,
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
            .datapointAmount = datapointAmount,
            .k = k
        };
        pthread_create(&threads[i], NULL, euclideanThreadHandler768D, &threadArgs[i]);
    }

    for (size_t i = 0; i < coreAmount; ++i) {
        pthread_join(threads[i], NULL);
    }
}
