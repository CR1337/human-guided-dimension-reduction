<style>

</style>

<template>
    <div id="canvas"></div>
</template>

<script>
import P5 from 'p5';
import { Quadtree } from '@/util/quadtree.js';

export default {
    name: 'Canvas',
    props: [
        'datapoints', 'k', 'showNeighbors', 'distanceMetric'
    ],
    data() {
        return {
            p5canvas: null,
            p5: null,

            xTranslation: null,
            yTranslation: null,
            scaling: null,

            quadtree: null,
            selectedPointIndex: null,
            hoveredPointIndex: null,

            knnIndices: [],

            // CONSTANTS
            canvasWidth: 640,
            canvasHeight: 640,
            landmarkSize: 10,
            pointSize: 5,
            zoomSpeed: 0.1,
            margin: 0.1,

            fills: [[0, 0, 255], [255, 255, 0]],
            hoveredFill: [0, 255, 0],
            selectedFill: [255, 0, 0],
            neighborStroke: [255, 0, 0],
            landmarkStroke: [0, 255, 255]
        }
    },
    methods: {
        datapointsUpdated() {
            let sumX = 0.0;
            let sumY = 0.0;
            let minX = Infinity;
            let maxX = -Infinity;
            let minY = Infinity;
            let maxY = -Infinity;
            for (const datapoint of this.datapoints) {
                const x = datapoint.position[0];
                const y = datapoint.position[1];
                sumX += x;
                sumY += y;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }

            const meanX = sumX / this.datapoints.length;
            const meanY = sumY / this.datapoints.length;

            this.xTranslation = -meanX + 0.5;
            this.yTranslation = -meanY + 0.5;

            if (this.canvasWidth < this.canvasHeight) {
                this.scaling = (this.canvasWidth * (1.0 - this.margin)) / (maxX - minX);
            } else {
                this.scaling = (this.canvasHeight * (1.0 - this.margin)) / (maxY - minY);
            }

            this.quadtree = new Quadtree(minX, minY, maxX, maxY);
            for (const [i, datapoint] of this.datapoints.entries()) {
                const x = datapoint.position[0];
                const y = datapoint.position[1];
                this.quadtree.insert(x, y, i);
            }

            this.changeTransformation(this.p5);
        },

        changeTransformation(p5) {
            p5.translate(this.xTranslation, this.yTranslation);
            p5.scale(this.scaling);
        },

        datapointIndexAtMouse(p5) {
            if (this.quadtree == null) {
                return null;
            }

            const mx = p5.mouseX / this.scaling - this.xTranslation;
            const my = p5.mouseY / this.scaling - this.yTranslation;
            const foundIndices = this.quadtree.lookup(mx, my);

            for (const foundIndex of foundIndices) {
                const datapoint = this.datapoints[foundIndex];
                const px = (datapoint.position[0] + this.xTranslation) * this.scaling;
                const py = (datapoint.position[1] + this.yTranslation) * this.scaling;
                if (p5.dist(p5.mouseX, p5.mouseY, px, py) < this.pointSize) {
                    return foundIndex;
                }
            }

            return null;
        },

        drawPoints(p5) {
            // Draw "normal" points
            for (const [i, datapoint] of this.datapoints.entries()) {
                if ([this.selectedPointIndex, this.hoveredPointIndex].includes(i)) continue;
                if (this.knnIndices.includes(i)) continue

                let size, shape, stroke;
                if (datapoint.is_landmark) {
                    size = this.landmarkSize;
                    shape = 'square';
                    stroke = this.landmarkStroke;
                } else {
                    size = this.pointSize;
                    shape = 'circle';
                    stroke = null;
                }

                this.drawPoint(
                    p5,
                    datapoint.position[0], datapoint.position[1],
                    size, this.fills[datapoint.label], stroke, shape
                );
            }

            // Draw neighbor connections
            if (this.showNeighbors) {
                p5.stroke(this.neighborStroke);
                for (const neighborIndex of this.knnIndices) {
                    const datapoint = this.datapoints[neighborIndex];
                    p5.line(
                        (datapoint.position[0] + this.xTranslation) * this.scaling,
                        (datapoint.position[1] + this.yTranslation) * this.scaling,
                        (this.datapoints[this.selectedPointIndex].position[0] + this.xTranslation) * this.scaling,
                        (this.datapoints[this.selectedPointIndex].position[1] + this.yTranslation) * this.scaling
                    );
                }
            }

            // Draw neighbors
            for (let i of this.knnIndices) {
                const datapoint = this.datapoints[i];

                let size, shape, stroke;
                if (datapoint.is_landmark) {
                    size = this.landmarkSize;
                    shape = 'square';
                    stroke = this.landmarkStroke;
                } else {
                    size = this.pointSize;
                    shape = 'circle';
                    stroke = null;
                }

                stroke = (this.showNeighbors) ? this.neighborStroke : stroke;

                this.drawPoint(
                    p5,
                    datapoint.position[0], datapoint.position[1],
                    size, this.fills[datapoint.label], stroke, shape
                );
            }

            // Draw selected point
            if (this.selectedPointIndex != null) {
                const datapoint = this.datapoints[this.selectedPointIndex];

                let size, shape, stroke;
                if (datapoint.is_landmark) {
                    size = this.landmarkSize;
                    shape = 'square';
                    stroke = this.landmarkStroke;
                } else {
                    size = this.pointSize;
                    shape = 'circle';
                    stroke = null;
                }

                this.drawPoint(
                    p5,
                    datapoint.position[0], datapoint.position[1],
                    size, this.selectedFill, stroke, shape
                );
            }

            // Draw hovered point
            if (this.hoveredPointIndex != null) {
                const datapoint = this.datapoints[this.hoveredPointIndex];

                let size, shape, stroke;
                if (datapoint.is_landmark) {
                    size = this.landmarkSize;
                    shape = 'square';
                    stroke = this.landmarkStroke;
                } else {
                    size = this.pointSize;
                    shape = 'circle';
                    stroke = null;
                }

                this.drawPoint(
                    p5,
                    datapoint.position[0], datapoint.position[1],
                    size, this.hoveredFill, stroke, shape
                );
            }
        },

        drawPoint(p5, x, y, size, fill, stroke, shape) {
            p5.fill(fill);
            (stroke == null) ? p5.noStroke() : p5.stroke(stroke);
            if (shape == 'circle')
                p5.circle(
                    (x + this.xTranslation) * this.scaling,
                    (y + this.yTranslation) * this.scaling,
                    size
                );
            else if (shape == 'square')
                p5.square(
                    (x + this.xTranslation) * this.scaling,
                    (y + this.yTranslation) * this.scaling,
                    size
                );
        },

        calculateCosineDistance(datapointA, dataPointB) {
            return Math.abs(datapointA.cosine - dataPointB.cosine);
        },

        calculateRealCosineDistance(datapointA, dataPointB) {
            const a = datapointA.position;
            const b = dataPointB.position;
            const dotProduct = a[0] * b[0] + a[1] * b[1];
            const aLength = Math.sqrt(a[0] * a[0] + a[1] * a[1]);
            const bLength = Math.sqrt(b[0] * b[0] + b[1] * b[1]);
            return 1.0 - (dotProduct / (aLength * bLength));
        },

        findKNearestNeighbors(datapoint) {
            if (this.distanceMetric == 'euclidean') {
                return this.quadtree.findKNearestNeighbors(
                    datapoint.position[0],
                    datapoint.position[1],
                    this.k
                );
            } else if (this.distanceMetric == 'cosine') {
                // FIXME: The result doesn't look right
                const neighborIndices = [];
                const length = this.datapoints.length;

                const cosines = [];
                for (const [i, neighbor] of this.datapoints.entries()) {
                    cosines.push({
                        index: i,
                        cosine: this.calculateRealCosineDistance(datapoint, neighbor)
                    });
                }
                cosines.sort((a, b) => a.cosine - b.cosine);
                neighborIndices.push(...cosines.slice(1, this.k + 1).map(c => c.index));


                //let leftIndex = this.selectedPointIndex - 1;
                //let rightIndex = this.selectedPointIndex + 1;
                //let leftEdgeReached = leftIndex < 0;
                //let rightEdgeReached = rightIndex >= length;
//
                //for (let i = 0; i < this.k; ++i) {
                //    if (leftEdgeReached && rightEdgeReached) break;
//
                //    if (leftEdgeReached) {
                //        neighborIndices.push(rightIndex++);
                //        rightEdgeReached = rightIndex >= length;
                //    } else if (rightEdgeReached) {
                //        neighborIndices.push(leftIndex--);
                //        leftEdgeReached = leftIndex < 0;
                //    } else {
                //        if (this.calculateCosineDistance(datapoint, this.datapoints[leftIndex]) < this.calculateCosineDistance(datapoint, this.datapoints[rightIndex])) {
                //            neighborIndices.push(leftIndex--);
                //            leftEdgeReached = leftIndex < 0;
                //        } else {
                //            neighborIndices.push(rightIndex++);
                //            rightEdgeReached = rightIndex >= length;
                //        }
                //    }
                //}

                return neighborIndices;
            }
        },

        // P5 FUNCTIONS
        setup(p5) {
            p5.createCanvas(this.canvasWidth, this.canvasHeight);
            p5.noStroke();
            p5.rectMode(p5.CENTER);
        },
        draw(p5) {
            p5.background(0);
            this.drawPoints(p5);
        },
        mouseDragged(p5, event) {
            this.xTranslation += event.movementX / this.scaling;
            this.yTranslation += event.movementY / this.scaling;
            this.changeTransformation(p5);
        },
        mouseMoved(p5, event) {
            this.hoveredPointIndex = this.datapointIndexAtMouse(p5);
            this.$emit('hoveredPointIndexChanged', this.hoveredPointIndex);
        },
        mouseWheel(p5, event) {
            const zoom = -event.delta * this.zoomSpeed;
            this.scaling += zoom;
            this.changeTransformation(p5);
        },
        mouseReleased(p5) {
            this.selectedPointIndex = this.datapointIndexAtMouse(p5);
            this.$emit('selectedPointIndexChanged', this.selectedPointIndex);
            if (this.selectedPointIndex == null) {
                this.knnIndices = [];
                return;
            }
            const datapoint = this.datapoints[this.selectedPointIndex];
            this.knnIndices = this.findKNearestNeighbors(datapoint);
        }
    },
    mounted() {
        const p5Script = p5 => {
            p5.setup = () => { this.p5 = p5; this.setup(p5); };
            p5.draw = () => { this.draw(p5); };
            p5.mouseReleased = () => { this.mouseReleased(p5); };
            p5.mouseDragged = (event) => { this.mouseDragged(p5, event); };
            p5.mouseMoved = (event) => { this.mouseMoved(p5, event); }
            p5.mouseWheel = (event) => { this.mouseWheel(p5, event); };
        }
        this.p5canvas = new P5(p5Script, 'canvas');
    }
}
</script>
