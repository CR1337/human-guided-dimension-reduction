#ifndef __QUDTREE_HPP__
#define __QUDTREE_HPP__

#include <cmath>
#include <array>
#include <vector>
#include <queue>

#include "types.hpp"

class QuadtreePoint {
public:
    float x;
    float y;

    constexpr static float EPSILON = 0.0000001f;

    QuadtreePoint() {};
    QuadtreePoint(float x, float y) : x(x), y(y) {};

    bool equals(QuadtreePoint *other);
    QuadtreePoint * center(QuadtreePoint *other);
    float distance(QuadtreePoint *other);
    float boundingBoxDistance(float minX, float minY, float maxX, float maxY);
};

class KnnHeapEntry {
public:
    QuadtreePoint *point;
    float distance;
    Index value;

    KnnHeapEntry() {};
    KnnHeapEntry(QuadtreePoint *point, float distance, Index value) : point(point), distance(distance), value(value) {};

    bool equals(KnnHeapEntry *other);
};

struct {
    bool operator()(const KnnHeapEntry &a, const KnnHeapEntry &b) const {
        return a.distance > b.distance;
    }
} KnnHeapEntryComparator;

class KnnHeap {
public:
    size_t k;
    std::priority_queue<KnnHeapEntry, std::vector<KnnHeapEntry>, decltype(KnnHeapEntryComparator)> q;

    KnnHeap(size_t k) : k(k) {
        this->q = std::priority_queue<KnnHeapEntry, std::vector<KnnHeapEntry>, decltype(KnnHeapEntryComparator)>(KnnHeapEntryComparator);
    };

    bool isEmpty();
    bool isFull();
    const KnnHeapEntry *top();
    bool pushOrReject(KnnHeapEntry *entry) ;
    void entries(std::vector<KnnHeapEntry> *entries);
    void firstKValues(std::vector<Index> *values);
};

class QuadtreeNode {
public:
    QuadtreePoint *leftTop;
    QuadtreePoint *rightBottom;

    QuadtreeNode() {};
    QuadtreeNode(QuadtreePoint *leftTop, QuadtreePoint *rightBottom) : leftTop(leftTop), rightBottom(rightBottom) {};

    virtual bool isEmpty() = 0;
    virtual QuadtreeNode * insert(QuadtreePoint *point, Index value) = 0;
    virtual bool findNearestNeighbors(QuadtreePoint *point, KnnHeap *heap) = 0;

};

class QuadtreeLeafNode : public QuadtreeNode {
public:
    QuadtreePoint *point;
    std::vector<Index> values;

    QuadtreeLeafNode() {};
    QuadtreeLeafNode(QuadtreePoint *leftTop, QuadtreePoint *rightBottom) : QuadtreeNode(leftTop, rightBottom) {
        this->point = nullptr;
        this->values = std::vector<Index>();
    };

    bool isEmpty();
    QuadtreeNode * insert(QuadtreePoint *point, Index value);
    bool findNearestNeighbors(QuadtreePoint *point, KnnHeap *heap);
};

class QuadtreeInnerNode : public QuadtreeNode {
public:
    QuadtreePoint *center;
    std::array<QuadtreeNode*, 4> children;

    QuadtreeInnerNode() {};
    QuadtreeInnerNode(QuadtreePoint *leftTop, QuadtreePoint *rightBottom) : QuadtreeNode(leftTop, rightBottom) {
        this->center = leftTop->center(rightBottom);
        this->children[0] = new QuadtreeLeafNode(leftTop, center);
        this->children[1] = new QuadtreeLeafNode(new QuadtreePoint(this->center->x, leftTop->y), new QuadtreePoint(rightBottom->x, this->center->y));
        this->children[2] = new QuadtreeLeafNode(new QuadtreePoint(leftTop->x, this->center->y), new QuadtreePoint(this->center->x, rightBottom->y));
        this->children[3] = new QuadtreeLeafNode(center, rightBottom);
    }

    bool isEmpty();
    size_t getChildIndexFor(QuadtreePoint *point);
    QuadtreeInnerNode * insert(QuadtreePoint *point, Index value);
    bool findNearestNeighbors(QuadtreePoint *point, KnnHeap *heap);
};

class Quadtree {
public:
    QuadtreePoint *leftTop;
    QuadtreePoint *rightBottom;
    QuadtreeNode *root;
    size_t elementCount;

    Quadtree() {};
    Quadtree(float minX, float minY, float maxX, float maxY) {
        this->leftTop = new QuadtreePoint(minX, minY);
        this->rightBottom = new QuadtreePoint(maxX, maxY);
        this->root = new QuadtreeLeafNode(this->leftTop, this->rightBottom);
        this->elementCount = 0;
    };

    bool isEmpty();
    bool insert(float x, float y, Index value);
    void findNearestNeighbors(float x, float y, size_t k, std::vector<Index> *values) const;
};

#endif // __QUDTREE_HPP__
