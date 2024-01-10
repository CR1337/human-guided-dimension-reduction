import { PriorityQueue } from '@datastructures-js/priority-queue';

class QuadtreePoint {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    equals(other) {
        return this.x == other.x && this.y == other.y;
    }

    center(other) {
        return new QuadtreePoint(
            (this.x + other.x) / 2, (this.y + other.y) / 2
        );
    }

    distance(other) {
        return Math.sqrt(
            Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2)
        );
    }

    boundingBoxDistance(minX, minY, maxX, maxY) {
        let targetX = this.x;
        if (this.x <= minX) {
            targetX = minX;
        } else if (this.x >= maxX) {
            targetX = maxX;
        }

        let targetY = this.y;
        if (this.y <= minY) {
            targetY = minY;
        } else if (this.y >= maxY) {
            targetY = maxY;
        }

        return this.distance(new QuadtreePoint(targetX, targetY));
    }
}

class QuadtreeDeleteResult {
    constructor(values, node, deleted) {
        this.values = values;
        this.node = node;
        this.deleted = deleted;
    }
}

class QuadtreeNode {
    constructor(leftTop, rightBottom) {
        this.leftTop = leftTop;
        this.rightBottom = rightBottom;
    }
}

class QuadtreeInnerNode extends QuadtreeNode {
    constructor(leftTop, rightBottom) {
        super(leftTop, rightBottom);
        this.center = leftTop.center(rightBottom);
        this.children = [
            new QuadtreeLeafNode(this.leftTop, this.center),
            new QuadtreeLeafNode(
                new QuadtreePoint(this.center.x, this.leftTop.y),
                new QuadtreePoint(this.rightBottom.x, this.center.y)
            ),
            new QuadtreeLeafNode(
                new QuadtreePoint(this.leftTop.x, this.center.y),
                new QuadtreePoint(this.center.x, this.rightBottom.y)
            ),
            new QuadtreeLeafNode(this.center, this.rightBottom),
        ]
    }

    isEmpty() { return false; }

    getChildIndexFor(point) {
        if (point.y <= this.center.y) {
            return (point.x <= this.center.x) ? 0 : 1;
        } else {
            return (point.x <= this.center.x) ? 2 : 3;
        }
    }

    lookup(point) {
        return this.children[this.getChildIndexFor(point)].lookup(point);
    }

    insert(point, value) {
        const childIndex = this.getChildIndexFor(point)
        const child = this.children[childIndex];
        const newNode = child.insert(point, value);
        this.children[childIndex] = newNode;
        return this;
    }

    delete(point) {
        const childIndex = this.getChildIndexFor(point);
        const child = this.children[childIndex];
        const deleteResult = child.delete(point);

        this.children[childIndex] = deleteResult.node;

        if (this.children.every((child) => child.isEmpty())) {
            const leafNode = new QuadtreeLeafNode(
                this.leftTop, this.rightBottom
            );
            return new QuadtreeDeleteResult(
                deleteResult.value, leafNode, deleteResult.deleted
            );
        }

        return new QuadtreeDeleteResult(
            deleteResult.values, this, deleteResult.deleted
        );
    }

    findNearestNeighbors(point, heap) {
        const childIndex = this.getChildIndexFor(point);
        this.children[childIndex].findNearestNeighbors(point, heap);
        for (let i = 0; i < 4; ++i) {
            if (i == childIndex) continue;
            const child = this.children[i];
            if (!heap.isFull()) {
                child.findNearestNeighbors(point, heap);
            } else {
                const minNodeDistance = point.boundingBoxDistance(
                    child.leftTop.x, child.leftTop.y,
                    child.rightBottom.x, child.rightBottom.y
                );
                if (minNodeDistance < heap.top().distance) {
                    child.findNearestNeighbors(point, heap);
                }
            }
        }
        return true;
    }
}

class QuadtreeLeafNode extends QuadtreeNode {
    constructor(leftTop, rightBottom) {
        super(leftTop, rightBottom);
        this.point = null;
        this.values = []
    }

    isEmpty() {
        return this.values.length == 0;
    }

    lookup(point) {
        return this.values;
    }

    insert(point, value) {
        if (this.isEmpty()) {
            this.point = point;
            this.values.push(value);
        } else if (point.equals(this.point)) {
            this.values.push(value);
        } else {
            let newNode = new QuadtreeInnerNode(
                this.leftTop, this.rightBottom
            );
            for (const val of this.values) {
                newNode.insert(this.point, val);
            }
            newNode.insert(point, value);
            return newNode;
        }
        return this;
    }

    delete(point) {
        this.point = null
        const result = new QuadtreeDeleteResult(
            this.values, this, !this.isEmpty()
        );
        this.values = [];
        return result;
    }

    findNearestNeighbors(point, heap) {
        for (const value of this.values) {
            const distance = point.distance(this.point);
            heap.pushOrReject(new KNNHeapEntry(this.point, distance, value));
        }
        return true;
    }
}

class KNNHeapEntry {
    constructor(point, distance, value) {
        this.point = point;
        this.distance = distance;
        this.value = value;
    }

    equals(other) {
        return this.point.equals(other.point);
    }
}

class KNNHeap {
    constructor(k) {
        this.k = k;
        this.q = new PriorityQueue((a, b) => {
            if (a.distance < b.distance) return 1;
            if (a.distance > b.distance) return -1;
            return 0;
        });
    }

    isEmpty() {
        return this.q.isEmpty();
    }

    isFull() {
        return this.q.size() == this.k + 1;
    }

    top() {
        return this.q.front();
    }

    pushOrReject(entry) {
        if (this.q.size() < this.k + 1) {
            this.q.enqueue(entry);
            return true;
        } else if (entry.distance < this.top().distance) {
            this.q.dequeue();
            this.q.enqueue(entry);
            return true;
        } else {
            return false;
        }
    }

    entries() {
        return this.q.toArray();
    }

    firstKValues() {
        return this.entries().map((entry) => entry.value).flat().slice(0, this.k);
    }
}

export class Quadtree {
    constructor(minX, minY, maxX, maxY) {
        this.leftTop = new QuadtreePoint(minX, minY);
        this.rightBottom = new QuadtreePoint(maxX, maxY);
        this.root = new QuadtreeLeafNode(this.leftTop, this.rightBottom);
        this.elementCount = 0;
    }

    isEmpty() {
        return this.elementCount == 0;
    }

    getElementCount() {
        return this.elementCount;
    }

    checkRange(x, y) {
        if (x < this.minX) return false;
        if (x > this.maxX) return false;
        if (y < this.minY) return false;
        if (y > this.maxY) return false;
        return true;
    }

    lookup(x, y) {
        if (!this.checkRange(x, y)) return [];
        const point = new QuadtreePoint(x, y);
        return this.root.lookup(point);
    }

    insert(x, y, value) {
        if (!this.checkRange(x, y)) return false;
        const point = new QuadtreePoint(x, y);
        const newRoot = this.root.insert(point, value);
        if (newRoot != null) this.root = newRoot;
        this.elementCount++;
        return true;
    }

    delete(x, y) {
        if (!this.checkRange(x, y)) return [];
        const point = new QuadtreePoint(x, y);
        const deleteResult = this.root.delete(point);
        this.root = deleteResult.node;
        if (deleteResult.deleted) this.elementCount--;
        return deleteResult.values;
    }

    findKNearestNeighbors(x, y, k) {
        if (!this.checkRange(x, y)) return [];
        const point = new QuadtreePoint(x, y);
        const heap = new KNNHeap(k);
        this.root.findNearestNeighbors(point, heap);
        return heap.firstKValues();
    }
}
