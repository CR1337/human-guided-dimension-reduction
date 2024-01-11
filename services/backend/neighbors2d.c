#include <stdio.h>
#include <math.h>
#include <sys/shm.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/ipc.h>
#include <stdint.h>
#include <string.h>
#include <sys/sysinfo.h>
#define TEMP __USE_XOPEN2K
#define __USE_XOPEN2K 600  // Required for pthread_barrier_t
#include <pthread.h>
#define __USE_XOPEN2K TEMP
#undef TEMP

#define ARGUMENT_COUNT (5)
#define ARGUMENT_BASE (10)
#define SHARED_MEMORY_ACCESS_FLAGS (0666)

#define EUCLIDEAN_DISTNACE_METRIC (0)
#define COSINE_DISTNACE_METRIC (1)

typedef uint64_t DistanceMetric;

#pragma pack(1)
typedef struct {
    float x;
    float y;
} Position;

#pragma pack(1)
typedef struct {
    uint32_t index;
    float distance;
} DistanceIndexPair;

int indexDistancePairCompare(const void *a, const void *b) {
    const DistanceIndexPair *distanceIndexPairA = (DistanceIndexPair*)a;
    const DistanceIndexPair *distanceIndexPairB = (DistanceIndexPair*)b;
    const float difference = distanceIndexPairA->distance - distanceIndexPairB->distance;
    return (difference > 0.0f) - (difference < 0.0f);
}

typedef struct {
    size_t offset;
    Position *positions;
    DistanceIndexPair *distanceIndexPairs;
    DistanceIndexPair *buffer;
    DistanceMetric distanceMetric;
    size_t coreAmount;
    size_t datapointAmount;
    size_t k;
} ThreadArgs;

void printError(const char *message) {
    fprintf(stderr, "%s\n", message);
}

inline float dot(const Position *a, const Position *b) {
    return a->x * b->x + a->y * b->y;
}

inline float norm(const Position *a) {
    return sqrt(dot(a, a));
}

inline void subtract(const Position *a, const Position *b, Position *result) {
    result->x = a->x - b->x;
    result->y = a->y - b->y;
}

inline float euclideanDistance(const Position *a, const Position *b) {
    Position difference;
    subtract(a, b, &difference);
    return norm(&difference);
}

inline float cosineDistance(const Position *a, const Position *b) {
    return 1.0f - dot(a, b) / (norm(a) * norm(b));
}

bool parseArguments(
    int argc,
    char *argv[],
    key_t *sharedMemoryKey,
    size_t *sharedMemorySize,
    DistanceMetric *distanceMetric,
    size_t *datapointAmount,
    size_t *k
) {
    if (argc - 1 != ARGUMENT_COUNT) return false;
    errno = 0;
    *sharedMemoryKey = strtoll(argv[1], NULL, ARGUMENT_BASE);
    if (errno != 0) return false;
    *sharedMemorySize = strtoull(argv[2], NULL, ARGUMENT_BASE);
    if (errno != 0) return false;
    *distanceMetric = strtoull(argv[3], NULL, ARGUMENT_BASE);
    if (errno != 0) return false;
    if (*distanceMetric != EUCLIDEAN_DISTNACE_METRIC && *distanceMetric != COSINE_DISTNACE_METRIC) return false;
    *datapointAmount = strtoull(argv[4], NULL, ARGUMENT_BASE);
    if (errno != 0) return false;
    *k = strtoull(argv[5], NULL, ARGUMENT_BASE);
    if (errno != 0) return false;
    return true;
}

bool attachSharedMemory(key_t key, size_t size, void **sharedMemory) {
    int sharedMemoryId = shmget(key, size, SHARED_MEMORY_ACCESS_FLAGS);
    if (sharedMemoryId == -1) return false;
    *sharedMemory = shmat(sharedMemoryId, NULL, 0);
    if (*sharedMemory == (void*)-1) return false;
    return true;
}

bool detachSharedMemory(void *sharedMemory) {
    return shmdt(sharedMemory) != -1;
}

void * threadHandler(void *args) {
    const ThreadArgs *threadArgs = (ThreadArgs*)args;
    const size_t start = threadArgs->offset;
    const Position *positions = threadArgs->positions;
    DistanceIndexPair *distanceIndexPairs = threadArgs->distanceIndexPairs;
    DistanceIndexPair *buffer = threadArgs->buffer;
    const DistanceMetric distanceMetric = threadArgs->distanceMetric;
    const size_t coreAmount = threadArgs->coreAmount;
    const size_t datapointAmount = threadArgs->datapointAmount;
    const size_t k = threadArgs->k;

    size_t end = (
        start
        + (datapointAmount / coreAmount)
        + (datapointAmount % coreAmount != 0)
    );
    if (end > datapointAmount) end = datapointAmount;

    switch (distanceMetric) {
        case EUCLIDEAN_DISTNACE_METRIC:
            for (size_t i = start; i < end; ++i) {
                const size_t rowOffset = i * datapointAmount;

                for (size_t j = 0; j < datapointAmount; ++j) {
                    buffer[rowOffset + j] = (DistanceIndexPair) {
                        .index = j,
                        .distance = euclideanDistance(positions + i, positions + j)
                    };
                }
            }
            break;
        case COSINE_DISTNACE_METRIC:
            for (size_t i = start; i < end; ++i) {
                const size_t rowOffset = i * datapointAmount;

                for (size_t j = 0; j < datapointAmount; ++j) {
                    buffer[rowOffset + j] = (DistanceIndexPair) {
                        .index = j,
                        .distance = cosineDistance(positions + i, positions + j)
                    };
                }
            }
            break;
    }

    size_t itemsToCopy = (k + 1 > datapointAmount) ? datapointAmount : k + 1;

    for (size_t i = start; i < end; ++i) {
        const size_t bufferRowOffset = i * datapointAmount;
        const size_t distanceIndexPairsRowOffset = i * itemsToCopy;
        qsort(
            &buffer[bufferRowOffset],
            datapointAmount,
            sizeof(DistanceIndexPair),
            indexDistancePairCompare
        );
        memcpy(
            &distanceIndexPairs[distanceIndexPairsRowOffset],
            &buffer[bufferRowOffset],
            itemsToCopy * sizeof(DistanceIndexPair)
        );
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    key_t sharedMemoryKey;
    size_t sharedMemorySize;
    DistanceMetric distanceMetric;
    size_t datapointAmount;
    size_t k;
    if (!parseArguments(
        argc, argv,
        &sharedMemoryKey, &sharedMemorySize,
        &distanceMetric, &datapointAmount, &k
    )) {
        printError("Invalid arguments");
        return EXIT_FAILURE;
    }

    void *sharedMemory;
    if (!attachSharedMemory(sharedMemoryKey, sharedMemorySize, &sharedMemory)) {
        printError("Failed to attach shared memory");
        return EXIT_FAILURE;
    }

    Position *positions = (Position*)sharedMemory;
    DistanceIndexPair *distanceIndexPairs = (DistanceIndexPair*)(positions + datapointAmount);

    DistanceIndexPair *buffer = (DistanceIndexPair*)malloc(
        datapointAmount * datapointAmount * sizeof(DistanceIndexPair)
    );
    if (buffer == NULL) {
        printError("Failed to allocate buffer");
        return EXIT_FAILURE;
    }

    size_t coreAmount = get_nprocs();
    if (coreAmount > datapointAmount) coreAmount = datapointAmount;
    pthread_t threads[coreAmount];
    ThreadArgs threadArgs[coreAmount];

    for (size_t i = 0; i < coreAmount; ++i) {
        threadArgs[i] = (ThreadArgs) {
            .offset = i * (
                (datapointAmount / coreAmount)
                + (datapointAmount % coreAmount != 0)
            ),
            .positions = positions,
            .distanceIndexPairs = distanceIndexPairs,
            .buffer = buffer,
            .distanceMetric = distanceMetric,
            .coreAmount = coreAmount,
            .datapointAmount = datapointAmount,
            .k = k
        };
        pthread_create(&threads[i], NULL, threadHandler, &threadArgs[i]);
    }

    for (size_t i = 0; i < coreAmount; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(buffer);
    if (!detachSharedMemory(sharedMemory)) {
        printError("Failed to detach shared memory");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
