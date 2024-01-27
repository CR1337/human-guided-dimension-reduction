#include <sys/shm.h>
#include <errno.h>

#include <iostream>

#include "types.hpp"
#include "euclidean.hpp"
#include "cosine.hpp"
#include "util.hpp"

#define ARGUMENT_COUNT (2)
#define ARGUMENT_BASE (10)
#define SHARED_MEMORY_ACCESS_FLAGS (0666)
#define SHARED_MEMORY_ERROR (-1)

bool parseArguments(
    int argc,
    char *argv[],
    key_t *sharedMemoryKey,
    size_t *sharedMemorySize
) {
    if (argc - 1 != ARGUMENT_COUNT) return false;
    errno = 0;
    *sharedMemoryKey = strtoll(argv[1], NULL, ARGUMENT_BASE);
    if (errno != 0) return false;
    *sharedMemorySize = strtoull(argv[2], NULL, ARGUMENT_BASE);
    if (errno != 0) return false;
    return true;
}

bool attachSharedMemory(key_t key, size_t size, void **sharedMemory) {
    int sharedMemoryId = shmget(key, size, SHARED_MEMORY_ACCESS_FLAGS);
    if (sharedMemoryId == SHARED_MEMORY_ERROR) return false;
    *sharedMemory = shmat(sharedMemoryId, NULL, 0);
    if (*sharedMemory == (void*)SHARED_MEMORY_ERROR) return false;
    return true;
}

bool detachSharedMemory(void *sharedMemory) {
    return shmdt(sharedMemory) != SHARED_MEMORY_ERROR;
}

bool computeNeighbors2D(
    DistanceMetric distanceMetric,
    size_t datapointAmount,
    void *sharedMemory
) {
    Position2D *positions = (Position2D*)sharedMemory;
    DistanceIndexPair *distanceIndexPairs = (DistanceIndexPair*)(positions + datapointAmount);
    Index *ranks = (Index*)(distanceIndexPairs + datapointAmount * datapointAmount);

    std::cout << "2 D:" << std::endl;
    std::cout << std::endl;
    std::cout << "size of  positions: " << (uint8_t*)distanceIndexPairs - (uint8_t*)positions << std::endl;
    std::cout << "size of  distanceIndexPairs: " << (uint8_t*)ranks - (uint8_t*)distanceIndexPairs << std::endl;
    std::cout << "size of  ranks: " << (uint8_t*)(ranks + datapointAmount * datapointAmount) - (uint8_t*)ranks << std::endl;
    std::cout << std::endl;

    switch (distanceMetric) {
        case EUCLIDEAN_DISTANCE_METRIC:
            findEuclideanNeighbors2D(
                positions,
                datapointAmount,
                distanceIndexPairs,
                ranks
            );
            break;
        case COSINE_DISTANCE_METRIC:
            findCosineNeighbors2D(
                positions,
                datapointAmount,
                distanceIndexPairs,
                ranks
            );
            break;
        default:
            return false;
    }

    return true;
}

bool computeNeighbors768D(
    DistanceMetric distanceMetric,
    size_t datapointAmount,
    void *sharedMemory
) {
    Position768D *positions = (Position768D*)sharedMemory;
    DistanceIndexPair *distanceIndexPairs = (DistanceIndexPair*)(positions + datapointAmount);
    Index *ranks = (Index*)(distanceIndexPairs + datapointAmount * datapointAmount);

    switch (distanceMetric) {
        case EUCLIDEAN_DISTANCE_METRIC:
            findEuclideanNeighbors768D(
                positions,
                datapointAmount,
                distanceIndexPairs,
                ranks
            );
            break;
        case COSINE_DISTANCE_METRIC:
            findCosineNeighbors768D(
                positions,
                datapointAmount,
                distanceIndexPairs,
                ranks
            );
            break;
        default:
            return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    key_t sharedMemoryKey;
    size_t sharedMemorySize;
    if (!parseArguments(
        argc, argv,
        &sharedMemoryKey, &sharedMemorySize
    )) {
        printError("Invalid arguments");
        return EXIT_FAILURE;
    }

    void *sharedMemory;
    if (!attachSharedMemory(sharedMemoryKey, sharedMemorySize, &sharedMemory)) {
        printError("Failed to attach shared memory");
        return EXIT_FAILURE;
    }

    Parameters *parameters = (Parameters*)sharedMemory;
    DistanceMetric distanceMetric = parameters->distanceMetric;
    Index datapointAmount = parameters->datapointAmount;
    DimensionCount dimensions = parameters->dimensions;

    void *originalSharedMemory = sharedMemory;
    sharedMemory = (void*)(parameters + 1);

    switch (dimensions) {
        case DIMENSIONS_2:
            if (!computeNeighbors2D(distanceMetric, datapointAmount, sharedMemory)) {
                printError("Invalid distance metric");
                return EXIT_FAILURE;
            }
            break;
        case DIMENSIONS_768:
            if (!computeNeighbors768D(distanceMetric, datapointAmount, sharedMemory)) {
                printError("Invalid distance metric");
                return EXIT_FAILURE;
            }
            break;
        default:
            printError("Invalid dimensions");
            return EXIT_FAILURE;
    }

    if (!detachSharedMemory(originalSharedMemory)) {
        printError("Failed to detach shared memory");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
