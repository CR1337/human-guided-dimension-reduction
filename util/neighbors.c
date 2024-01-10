// On Linux compile with: gcc neighbors.c -o neighbors -lm -lpthread -Ofast -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5 -march=native -ffast-math -flto

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/sysinfo.h>
#define TEMP __USE_XOPEN2K
#define __USE_XOPEN2K 600  // Required for pthread_barrier_t
#include <pthread.h>
#define __USE_XOPEN2K TEMP
#undef TEMP

#define DIMENSION_AMOUNT (768)
#define BYTES_PER_FLOAT (sizeof(float))
#define DATAPOINT_SIZE (DIMENSION_AMOUNT * BYTES_PER_FLOAT)

#define EMBEDDINGS_FILE_PATH "./volumes/data/imdb_embeddings.bin"
#define EUCLIDEAN_NEIGHBORS_FILE_PATH "./volumes/data/imdb_euclidean_neighbors.bin"
#define COSINE_NEIGHBORS_FILE_PATH "./volumes/data/imdb_cosine_neighbors.bin"

typedef struct {
    uint32_t index;
    float distance;
} IndexDistancePair;

int indexDistancePairCompare(const void *a, const void *b) {
    const IndexDistancePair *pairA = (IndexDistancePair*)a;
    const IndexDistancePair *pairB = (IndexDistancePair*)b;
    const float difference = pairA->distance - pairB->distance;
    return (difference > 0) - (difference < 0);
}

typedef struct {
    size_t offset;
    float *embeddings;
    IndexDistancePair *euclideanNeighbors;
    IndexDistancePair *cosineNeighbors;
    size_t coreAmount;
    size_t datapointAmount;
    size_t threadId;
    pthread_barrier_t *barrier;
} ThreadArgs;

float dot(const float *v1, const float *v2) {
    float result = 0.0;
    for (size_t i = 0; i < DIMENSION_AMOUNT; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

float norm(const float *v) {
    return sqrtf(dot(v, v));
}

void sub(const float *v1, const float *v2, float *result) {
    for (size_t i = 0; i < DIMENSION_AMOUNT; ++i) {
        result[i] = v1[i] - v2[i];
    }
}

float euclideanDistance(const float *v1, const float *v2) {
    float vectorBuffer[DATAPOINT_SIZE];
    sub(v1, v2, vectorBuffer);
    return norm(vectorBuffer);
}

float cosineDistance(const float *v1, const float *v2) {
    return dot(v1, v2) / (norm(v1) * norm(v2));
}

FILE * openEmbeddingsFile(size_t *fileSize) {
    FILE *file;

    if ((file = fopen(EMBEDDINGS_FILE_PATH, "rb")) == NULL) {
        perror("Error opening file!");
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        perror("Error seeking the end of the file");
        fclose(file);
        return NULL;
    }

    if ((*fileSize = ftell(file)) == -1) {
        perror("Error getting file size");
        fclose(file);
        return NULL;
    }

    rewind(file);
    return file;
}

FILE * openNeighborsFile(const char *filePath) {
    FILE *file;

    if ((file = fopen(filePath, "wb")) == NULL) {
        perror("Error opening file!");
        return NULL;
    }

    return file;
}

void * threadHandler(void *args) {
    const ThreadArgs *threadArgs = (ThreadArgs*)args;
    const size_t start = threadArgs->offset;
    const float *embeddings = threadArgs->embeddings;
    IndexDistancePair *euclideanNeighbors = threadArgs->euclideanNeighbors;
    IndexDistancePair *cosineNeighbors = threadArgs->cosineNeighbors;
    const size_t coreAmount = threadArgs->coreAmount;
    const size_t datapointAmount = threadArgs->datapointAmount;
    const size_t threadId = threadArgs->threadId;
    pthread_barrier_t *barrier = threadArgs->barrier;

    size_t end = (
        start
        + (datapointAmount / coreAmount)
        + (datapointAmount % coreAmount != 0)
    );
    if (end > datapointAmount) end = datapointAmount;

    printf("Thread %ld: %ld - %ld\n", threadId, start, end);
    pthread_barrier_wait(barrier);

    if (start == 0) {
        putchar('\n');
        for (size_t i = start; i < end; ++i) {
            printf("%ld / %ld\n", (i + 1) * coreAmount, end * coreAmount);
            const size_t rowOffset = i * datapointAmount;

            for (size_t j = 0; j < datapointAmount; ++j) {
                const size_t neighborIndex = rowOffset + j;
                const size_t embeddingIndexI = i * DIMENSION_AMOUNT;
                const size_t embeddingIndexJ = j * DIMENSION_AMOUNT;
                const float *embeddingI = &embeddings[embeddingIndexI];
                const float *embeddingJ = &embeddings[embeddingIndexJ];

                euclideanNeighbors[neighborIndex].index = j;
                euclideanNeighbors[neighborIndex].distance = euclideanDistance(
                    embeddingI, embeddingJ
                );

                cosineNeighbors[neighborIndex].index = j;
                cosineNeighbors[neighborIndex].distance = cosineDistance(
                    embeddingI, embeddingJ
                );
            }

            qsort(
                &euclideanNeighbors[rowOffset],
                datapointAmount,
                sizeof(IndexDistancePair),
                indexDistancePairCompare
            );
        }
    } else {
        for (size_t i = start; i < end; ++i) {
            const size_t rowOffset = i * datapointAmount;

            for (size_t j = 0; j < datapointAmount; ++j) {
                const size_t neighborIndex = rowOffset + j;
                const size_t embeddingIndexI = i * DIMENSION_AMOUNT;
                const size_t embeddingIndexJ = j * DIMENSION_AMOUNT;
                const float *embeddingI = &embeddings[embeddingIndexI];
                const float *embeddingJ = &embeddings[embeddingIndexJ];

                euclideanNeighbors[neighborIndex].index = j;
                euclideanNeighbors[neighborIndex].distance = euclideanDistance(
                    embeddingI, embeddingJ
                );

                cosineNeighbors[neighborIndex].index = j;
                cosineNeighbors[neighborIndex].distance = cosineDistance(
                    embeddingI, embeddingJ
                );
            }

            qsort(
                &euclideanNeighbors[rowOffset],
                datapointAmount,
                sizeof(IndexDistancePair),
                indexDistancePairCompare
            );
        }
    }
}

int main(int argc, char *argv[]) {

    // ------------------------------------------------------------------------
    printf("Reading embeddings from file...\n");

    size_t fileSize;
    FILE *embeddingsFile = openEmbeddingsFile(&fileSize);
    if (embeddingsFile == NULL) return EXIT_FAILURE;
    const size_t datapointAmount = fileSize / DATAPOINT_SIZE;

    float *embeddings = (float*)malloc(datapointAmount * DATAPOINT_SIZE);
    if (embeddings == NULL) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    const size_t readBytes = fread(
        embeddings, sizeof(int8_t), fileSize, embeddingsFile
    );
    if (readBytes != fileSize) {
        perror("Error reading the file");
        return EXIT_FAILURE;
    }
    fclose(embeddingsFile);

    // ------------------------------------------------------------------------

    printf("Allocating memory for neighbors...\n");

    IndexDistancePair *euclideanNeighbors = (IndexDistancePair*)malloc(
        sizeof(IndexDistancePair) * datapointAmount * datapointAmount
    );
    if (euclideanNeighbors == NULL) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    IndexDistancePair *cosineNeighbors = (IndexDistancePair*)malloc(
        sizeof(IndexDistancePair) * datapointAmount * datapointAmount
    );
    if (euclideanNeighbors == NULL) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------------

    printf("Starting threaded neighbor calculation...\n");

    const size_t coreAmount = get_nprocs();
    pthread_t threads[coreAmount];
    ThreadArgs threadArgs[coreAmount];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, coreAmount);
    putchar('\n');

    for (size_t i = 0; i < coreAmount; ++i) {
        threadArgs[i] = (ThreadArgs){
            .offset = i * (
                (datapointAmount / coreAmount)
                + (datapointAmount % coreAmount != 0)
            ),
            .embeddings = embeddings,
            .euclideanNeighbors = euclideanNeighbors,
            .cosineNeighbors = cosineNeighbors,
            .coreAmount = coreAmount,
            .datapointAmount = datapointAmount,
            .threadId = i,
            .barrier = &barrier
        };
        pthread_create(&threads[i], NULL, threadHandler, &threadArgs[i]);
    }

    for (size_t i = 0; i < coreAmount; ++i) {
        pthread_join(threads[i], NULL);
    }
    free(embeddings);
    putchar('\n');

    // ------------------------------------------------------------------------

    printf("Writing euclidean neighbors to file...\n");

    FILE *euclideanNeighborsFile = openNeighborsFile(
        EUCLIDEAN_NEIGHBORS_FILE_PATH
    );
    if (euclideanNeighborsFile == NULL) return EXIT_FAILURE;
    fwrite(
        euclideanNeighbors,
        sizeof(IndexDistancePair),
        datapointAmount * datapointAmount,
        euclideanNeighborsFile
    );
    fclose(euclideanNeighborsFile);

    // ------------------------------------------------------------------------

    printf("Writing cosine neighbors to file...\n");

    FILE *cosineNeighborsFile = openNeighborsFile(COSINE_NEIGHBORS_FILE_PATH);
    if (cosineNeighborsFile == NULL) return EXIT_FAILURE;
    fwrite(
        cosineNeighbors,
        sizeof(IndexDistancePair),
        datapointAmount * datapointAmount,
        cosineNeighborsFile
    );
    fclose(cosineNeighborsFile);

    // ------------------------------------------------------------------------

    free(euclideanNeighbors);
    free(cosineNeighbors);

    printf("Done!\n");

    return EXIT_SUCCESS;
}
