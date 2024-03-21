g++ neighbors/neighbors.cpp neighbors/euclidean.cpp neighbors/cosine.cpp neighbors/util.cpp neighbors/types.cpp^
    -o neighbors/neighbors -Wall -Wno-subobject-linkage -lm -lpthread ^
    -Ofast -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5 -march=native -ffast-math -flto
