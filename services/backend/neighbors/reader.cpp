#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unordered_set>

#include "types.hpp"

bool readIndex(Index *index, Index datapointAmount) {
    std::cin >> *index;
    if (*index >= datapointAmount) {
        std::cerr << "Invalid index" << std::endl;
        return false;
    }
    return true;
}

bool readIndexAndK(Index *index, Index *k, Index datapointAmount) {
    if (!readIndex(index, datapointAmount)) {
        return false;
    }
    std::cin >> *k;
    if (*k >= datapointAmount - *index) {
        std::cerr << "Invalid k" << std::endl;
        return false;
    }
    return true;
}

void printPosition(Position2D *positions2D, Position768D *positions768D, DimensionCount dimensions, Index index) {
    switch (dimensions) {
        case DIMENSIONS_2: {
            Position2D position2D = positions2D[index];
            std::cout << "(" << position2D.x << ", " << position2D.y << ")" << std::endl;
            break;
        }

        case DIMENSIONS_768: {
            Position768D position768D = positions768D[index];
            std::cout << "(";
            for (size_t i = 0; i < DIMENSIONS_768; ++i) {
                std::cout << position768D.values[i];
                if (i != DIMENSIONS_768 - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << ")" << std::endl;
            break;
        }
    }
}

void printDistances(DistanceIndexPair *distanceIndexPairs, Index datapointAmount, Index index, Index k) {
    float lastDistance = -1.0f;
    std::unordered_set<Index> seenIndices;
    for (size_t i = 0; i < k; ++i) {
        DistanceIndexPair distanceIndexPair = distanceIndexPairs[index * datapointAmount + i];
        std::cout << distanceIndexPair.index << "\t" << distanceIndexPair.distance;
        if (lastDistance > distanceIndexPair.distance) {
            std::cout << " (wrong order)";
        }
        if (seenIndices.find(distanceIndexPair.index) != seenIndices.end()) {
            std::cout << " (duplicate)";
        }
        std::cout << std::endl;
        lastDistance = distanceIndexPair.distance;
        seenIndices.insert(distanceIndexPair.index);
    }
    std::cout << std::endl;
}

void printRanks(Index *ranks, Index datapointAmount, Index index, Index k) {
    std::unordered_set<Index> seenIndices;
    for (size_t i = 0; i < k; ++i) {
        Index rank = ranks[index * datapointAmount + i];
        std::cout << rank;
        if (seenIndices.find(rank) != seenIndices.end()) {
            std::cout << " (duplicate)";
        }
        std::cout << std::endl;
        seenIndices.insert(rank);
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return EXIT_FAILURE;
    }
    const char *filename = argv[1];

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        std::cerr << "Failed to open file" << std::endl;
        return EXIT_FAILURE;
    }

    Parameters parameters;
    if (fread(&parameters, sizeof(Parameters), 1, file) != 1) {
        std::cerr << "Failed to read parameters" << std::endl;
        return EXIT_FAILURE;
    }

    DistanceMetric distanceMetric = parameters.distanceMetric;
    Index datapointAmount = parameters.datapointAmount;
    DimensionCount dimensions = parameters.dimensions;

    std::cout << "Distance metric: " << distanceMetric << std::endl;
    std::cout << "Datapoint amount: " << datapointAmount << std::endl;
    std::cout << "Dimensions: " << dimensions << std::endl;
    std::cout << std::endl;

    Position2D *positions2D = NULL;
    Position768D *positions768D = NULL;

    if (dimensions == DIMENSIONS_2) {
        positions2D = (Position2D*)malloc(sizeof(Position2D) * datapointAmount);
        if (positions2D == NULL) {
            std::cerr << "Failed to allocate memory" << std::endl;
            return EXIT_FAILURE;
        }
        if (fread(positions2D, sizeof(Position2D), datapointAmount, file) != datapointAmount) {
            std::cerr << "Failed to read positions" << std::endl;
            return EXIT_FAILURE;
        }
    } else if (dimensions == DIMENSIONS_768) {
        positions768D = (Position768D*)malloc(sizeof(Position768D) * datapointAmount);
        if (positions768D == NULL) {
            std::cerr << "Failed to allocate memory" << std::endl;
            return EXIT_FAILURE;
        }
        if (fread(positions768D, sizeof(Position768D), datapointAmount, file) != datapointAmount) {
            std::cerr << "Failed to read positions" << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        std::cerr << "Invalid dimensions" << std::endl;
        return EXIT_FAILURE;
    }

    DistanceIndexPair *distanceIndexPairs = (DistanceIndexPair*)malloc(sizeof(DistanceIndexPair) * datapointAmount * datapointAmount);
    if (distanceIndexPairs == NULL) {
        std::cerr << "Failed to allocate memory" << std::endl;
        return EXIT_FAILURE;
    }
    if (fread(distanceIndexPairs, sizeof(DistanceIndexPair), datapointAmount * datapointAmount, file) != datapointAmount * datapointAmount) {
        std::cerr << "Failed to read distance index pairs" << std::endl;
        return EXIT_FAILURE;
    }

    Index *ranks = (Index*)malloc(sizeof(Index) * datapointAmount * datapointAmount);
    if (ranks == NULL) {
        std::cerr << "Failed to allocate memory" << std::endl;
        return EXIT_FAILURE;
    }
    if (fread(ranks, sizeof(Index), datapointAmount * datapointAmount, file) != datapointAmount * datapointAmount) {
        std::cerr << "Failed to read ranks" << std::endl;
        return EXIT_FAILURE;
    }

    fclose(file);

    bool exit = false;
    while (!exit) {
        std::cout << ">";
        std::string command = "";
        std::cin >> command;

        Index index = 0;
        Index k = 0;

        switch (command[0]) {
            case 'x':
                exit = true;
                break;

            case 'p':
                if (readIndex(&index, datapointAmount)) {
                    printPosition(positions2D, positions768D, dimensions, index);
                }
                break;

            case 'd':
                if (readIndexAndK(&index, &k, datapointAmount)) {
                    printDistances(distanceIndexPairs, datapointAmount, index, k);
                }
                break;

            case 'r':
                if (readIndexAndK(&index, &k, datapointAmount)) {
                    printRanks(ranks, datapointAmount, index, k);
                }
                break;
        }
    }

    free(positions2D);
    free(positions768D);
    free(distanceIndexPairs);
    free(ranks);
    return EXIT_SUCCESS;
}