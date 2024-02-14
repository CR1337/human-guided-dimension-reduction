#include "quadtree.hpp"

bool QuadtreePoint::equals(QuadtreePoint *other) {
    return
        abs(this->x - other->x) < this->EPSILON
        && abs(this->y - other->y) < this->EPSILON;
}

QuadtreePoint * QuadtreePoint::center(QuadtreePoint *other) {
    return new QuadtreePoint((x + other->x) / 2, (y + other->y) / 2);
}

float QuadtreePoint::distance(QuadtreePoint *other) {
    return hypotf(x - other->x, y - other->y);
}

float QuadtreePoint::boundingBoxDistance(float minX, float minY, float maxX, float maxY) {
    float targetX = std::max(minX, std::min(x, maxX));
    float targetY = std::max(minY, std::min(y, maxY));
    QuadtreePoint target(targetX, targetY);
    return this->distance(&target);
}

bool KnnHeapEntry::equals(KnnHeapEntry *other) {
    return this->point->equals(other->point);
}

bool KnnHeap::isEmpty() {
    return this->q.empty();
}

bool KnnHeap::isFull() {
    return this->q.size() == this->k + 1;
}

const KnnHeapEntry * KnnHeap::top() {
    return &(this->q.top());
}

bool KnnHeap::pushOrReject(KnnHeapEntry *entry) {
    if (this->q.size() < this->k + 1) {
        this->q.push(*entry);
        return true;
    } else if (entry->distance < this->q.top().distance) {
        this->q.pop();
        this->q.push(*entry);
        return true;
    } else {
        return false;
    }
}

void KnnHeap::entries(std::vector<KnnHeapEntry> *entries) {
    while (!this->q.empty()) {
        entries->push_back(this->q.top());
        this->q.pop();
    }
}

void KnnHeap::firstKValues(std::vector<Index> *values) {
    std::vector<KnnHeapEntry> entries;
    this->entries(&entries);
    for (size_t i = 0; i < this->k; i++) {
        values->push_back(entries[i].value);
    }
}

bool QuadtreeLeafNode::isEmpty() { return this->values.empty(); }

QuadtreeNode * QuadtreeLeafNode::insert(QuadtreePoint *point, Index value) {
    if (this->isEmpty()) {
        this->point = point;
        this->values.push_back(value);
    } else if (point->equals(this->point)) {
        this->values.push_back(value);
    } else {
        QuadtreeInnerNode *newNode = new QuadtreeInnerNode(this->leftTop, this->rightBottom);
        for (Index &value : this->values) {
            newNode->insert(this->point, value);
        }
        newNode->insert(point, value);
        return newNode;
    }
    return this;
}

bool QuadtreeLeafNode::findNearestNeighbors(QuadtreePoint *point, KnnHeap *heap) {
    for (Index &value : this->values) {
        const float distance = this->point->distance(point);
        KnnHeapEntry *entry = new KnnHeapEntry(this->point, distance, value);
        heap->pushOrReject(entry);
    }
    return true;
}

bool QuadtreeInnerNode::isEmpty() { return false; }
size_t QuadtreeInnerNode::getChildIndexFor(QuadtreePoint *point) {
    if (point->y <= this->center->y) {
        return (point->x <= this->center->x) ? 0 : 1;
    } else {
        return (point->x <= this->center->x) ? 2 : 3;
    }
}

QuadtreeInnerNode * QuadtreeInnerNode::insert(QuadtreePoint *point, Index value) {
    const size_t childIndex = this->getChildIndexFor(point);
    QuadtreeNode *child = this->children[childIndex];
    QuadtreeNode *newChild = child->insert(point, value);
    this->children[childIndex] = newChild;
    return this;
}

bool QuadtreeInnerNode::findNearestNeighbors(QuadtreePoint *point, KnnHeap *heap) {
    const size_t childIndex = this->getChildIndexFor(point);
    this->children[childIndex]->findNearestNeighbors(point, heap);
    for (size_t i = 0; i < 4; i++) {
        if (i == childIndex) continue;
        const QuadtreeNode *child = this->children[i];
        if (!heap->isFull()) {
            const_cast<QuadtreeNode*>(child)->findNearestNeighbors(point, heap);
        } else {
            const float minNodeDistance = point->boundingBoxDistance(
                child->leftTop->x, child->leftTop->y,
                child->rightBottom->x, child->rightBottom->y
            );
            if (minNodeDistance < heap->top()->distance) {
                const_cast<QuadtreeNode*>(child)->findNearestNeighbors(point, heap);
            }
        }
    }
    return true;
}

bool Quadtree::isEmpty() { return this->elementCount == 0; }

bool Quadtree::insert(float x, float y, Index value) {
    QuadtreePoint *point = new QuadtreePoint(x, y);
    QuadtreeNode *newRoot = this->root->insert(point, value);
    this->root = newRoot;
    this->elementCount++;
    return true;
}

void Quadtree::findNearestNeighbors(float x, float y, size_t k, std::vector<Index> *values) const {
    QuadtreePoint *point = new QuadtreePoint(x, y);
    KnnHeap *heap = new KnnHeap(k);
    this->root->findNearestNeighbors(point, heap);
    heap->firstKValues(values);
}