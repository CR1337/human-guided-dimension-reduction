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

            landmarkIndices: [],

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

            this.landmarkIndices = [];
            this.quadtree = new Quadtree(minX, minY, maxX, maxY);
            for (const [i, datapoint] of this.datapoints.entries()) {
                const x = datapoint.position[0];
                const y = datapoint.position[1];
                this.quadtree.insert(x, y, i);
                if (datapoint.is_landmark) {
                    this.landmarkIndices.push(i);
                }
            }

            this.changeTransformation(this.p5);
        },

        changeTransformation(p5) {
            p5.translate(this.xTranslation, this.yTranslation);
            p5.scale(this.scaling);
        },

        datapointIndexAtMouse(p5) {
            for (const landmarkIndex of this.landmarkIndices) {
                const datapoint = this.datapoints[landmarkIndex];
                const px = (datapoint.position[0] + this.xTranslation) * this.scaling;
                const py = (datapoint.position[1] + this.yTranslation) * this.scaling;
                if (p5.dist(p5.mouseX, p5.mouseY, px, py) <= this.landmarkSize) {
                    return landmarkIndex;
                }
            }

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
                if (p5.dist(p5.mouseX, p5.mouseY, px, py) <= this.pointSize) {
                    return foundIndex;
                }
            }

            return null;
        },

        drawPoints(p5) {
            const landmarks = [];

            // Draw "normal" points
            for (const [i, datapoint] of this.datapoints.entries()) {
                if ([this.selectedPointIndex, this.hoveredPointIndex].includes(i)) continue;
                if (this.knnIndices.includes(i)) continue

                let size, shape, stroke;
                if (datapoint.is_landmark) {
                    landmarks.push({
                        position: [datapoint.position[0], datapoint.position[1]],
                        fill: this.fills[datapoint.label],
                        stroke: this.landmarkStroke
                    })
                    continue;
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
                    landmarks.push({
                        position: [datapoint.position[0], datapoint.position[1]],
                        fill: this.fills[datapoint.label],
                        stroke: (this.showNeighbors) ? this.neighborStroke : this.landmarkStroke
                    })
                    continue;
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
                let is_landmark = false;
                if (datapoint.is_landmark) {
                    landmarks.push({
                        position: [datapoint.position[0], datapoint.position[1]],
                        fill: this.selectedFill,
                        stroke: this.landmarkStroke
                    })
                    is_landmark = true;
                } else {
                    size = this.pointSize;
                    shape = 'circle';
                    stroke = null;
                }

                if (!is_landmark) {
                    this.drawPoint(
                        p5,
                        datapoint.position[0], datapoint.position[1],
                        size, this.selectedFill, stroke, shape
                    );
                }
            }

            // Draw hovered point
            if (this.hoveredPointIndex != null) {
                const datapoint = this.datapoints[this.hoveredPointIndex];

                let size, shape, stroke;
                let is_landmark = false;
                if (datapoint.is_landmark) {
                    landmarks.push({
                        position: [datapoint.position[0], datapoint.position[1]],
                        fill: this.hoveredFill,
                        stroke: this.landmarkStroke
                    })
                    is_landmark = true;
                } else {
                    size = this.pointSize;
                    shape = 'circle';
                    stroke = null;
                }

                if (!is_landmark) {
                    this.drawPoint(
                        p5,
                        datapoint.position[0], datapoint.position[1],
                        size, this.hoveredFill, stroke, shape
                    );
                }
            }

            // Draw landmarks
            p5.strokeWeight(2);
            for (const landmark of landmarks) {
                this.drawPoint(
                    p5,
                    landmark.position[0], landmark.position[1],
                    this.landmarkSize, landmark.fill, landmark.stroke, 'square'
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

        relativeAbsoluteAngle(datapointA, dataPointB) {
            let diff = Math.abs(datapointA.angle - dataPointB.angle);
            if (diff > Math.PI) diff = (2 * Math.PI) - diff;
            return Math.abs(diff);
        },

        findKNearestNeighbors(datapoint) {
            if (this.distanceMetric == 'euclidean') {
                return this.quadtree.findKNearestNeighbors(
                    datapoint.position[0],
                    datapoint.position[1],
                    this.k
                );
            } else if (this.distanceMetric == 'cosine') {
                const neighborIndices = [];
                const length = this.datapoints.length;

                let leftIndex = (this.selectedPointIndex ==0)
                    ? length - 1
                    : this.selectedPointIndex - 1;
                let rightIndex = (this.selectedPointIndex == length - 1)
                    ? 0
                    : this.selectedPointIndex + 1;

                for (let i = 0; i < this.k; ++i) {
                    if (this.relativeAbsoluteAngle(datapoint, this.datapoints[leftIndex]) < this.relativeAbsoluteAngle(datapoint, this.datapoints[rightIndex])) {
                        neighborIndices.push(leftIndex--);
                        if (leftIndex < 0) leftIndex = length - 1;
                    } else {
                        neighborIndices.push(rightIndex++);
                        if (rightIndex >= length) rightIndex = 0;
                    }
                }
                return neighborIndices;
            }
        },

        // P5 FUNCTIONS
        setup(p5) {
            p5.createCanvas(this.canvasWidth, this.canvasHeight);
            p5.frameRate(24);
            p5.noStroke();
            p5.rectMode(p5.CENTER);
        },
        draw(p5) {
            p5.background(0);
            this.drawPoints(p5);
            p5.fill('white');
            p5.noStroke();
            p5.text(p5.frameRate().toFixed(2) + " fps", 10, 10);
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
