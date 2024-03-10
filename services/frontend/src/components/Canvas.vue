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
        'datapoints', 'k', 'showNeighbors', 'distanceMetric',
        'coloring', 'metrics'
    ],
    data() {
        return {
            p5canvas: null,
            p5: null,

            xTranslation: 320,
            yTranslation: 320,
            scaling: 1,

            quadtree: null,
            selectedPointIndex: null,
            hoveredPointIndex: null,
            landmarkIndices: [],

            pointIsMoving: false,
            newPosition: null,

            knnIndices: [],

            needRerender: false,
            timeSinceLastFpsUpdate: 0,

            // datapointDrawingInformation: [],

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
            landmarkStroke: [0, 255, 255],

            colorScale: [[10, 47, 81], [10, 48, 82], [10, 49, 83], [11, 50, 83], [11, 50, 84], [11, 51, 84], [11, 52, 85], [11, 53, 86], [11, 54, 86], [11, 55, 87], [11, 56, 87], [11, 57, 88], [12, 58, 89], [12, 59, 89], [12, 59, 90], [12, 60, 90], [12, 61, 91], [12, 62, 92], [12, 63, 92], [12, 64, 93], [13, 65, 93], [13, 66, 94], [13, 67, 95], [13, 68, 95], [13, 69, 96], [13, 70, 96], [13, 71, 97], [13, 72, 97], [14, 73, 98], [14, 74, 99], [14, 75, 99], [14, 76, 100], [14, 77, 100], [14, 78, 101], [14, 79, 102], [14, 80, 102], [15, 81, 103], [15, 82, 103], [15, 83, 104], [15, 85, 104], [15, 86, 105], [15, 87, 106], [15, 88, 106], [16, 89, 107], [16, 90, 107], [16, 91, 108], [16, 92, 109], [16, 93, 109], [16, 94, 110], [16, 96, 110], [17, 97, 111], [17, 98, 111], [17, 99, 112], [17, 100, 113], [17, 101, 113], [17, 102, 114], [17, 104, 114], [17, 105, 115], [18, 106, 115], [18, 107, 116], [18, 108, 117], [18, 110, 117], [18, 111, 118], [18, 112, 118], [19, 113, 119], [19, 114, 119], [19, 116, 120], [19, 117, 121], [19, 118, 121], [19, 119, 122], [19, 120, 122], [20, 122, 123], [20, 123, 123], [20, 124, 124], [20, 124, 123], [20, 125, 123], [20, 126, 123], [20, 126, 123], [21, 127, 123], [21, 127, 123], [21, 128, 123], [21, 128, 122], [21, 129, 122], [21, 130, 122], [22, 130, 122], [22, 131, 122], [22, 131, 121], [22, 132, 121], [22, 132, 121], [22, 133, 121], [23, 133, 121], [23, 134, 120], [23, 135, 120], [23, 135, 120], [23, 136, 120], [23, 136, 119], [24, 137, 119], [24, 137, 119], [24, 138, 119], [24, 138, 118], [24, 139, 118], [24, 139, 118], [25, 140, 117], [25, 141, 117], [25, 141, 117], [25, 142, 117], [25, 142, 116], [25, 143, 116], [26, 143, 116], [26, 144, 115], [26, 144, 115], [26, 145, 115], [26, 145, 114], [26, 146, 114], [27, 147, 114], [27, 147, 113], [27, 148, 113], [27, 148, 112], [27, 149, 112], [27, 149, 112], [28, 150, 111], [28, 150, 111], [28, 151, 111], [28, 151, 110], [28, 152, 110], [29, 152, 109], [29, 153, 109], [29, 154, 108], [29, 154, 108], [30, 155, 108], [32, 156, 108], [33, 156, 108], [34, 157, 108], [36, 158, 107], [37, 158, 107], [38, 159, 107], [40, 160, 107], [41, 161, 107], [42, 161, 107], [44, 162, 107], [45, 163, 107], [46, 163, 107], [48, 164, 107], [49, 165, 107], [50, 166, 107], [52, 166, 107], [53, 167, 107], [54, 168, 107], [56, 168, 107], [57, 169, 107], [58, 170, 107], [60, 170, 107], [61, 171, 108], [62, 172, 108], [64, 173, 108], [65, 173, 108], [66, 174, 108], [68, 175, 108], [69, 175, 108], [70, 176, 109], [72, 177, 109], [73, 177, 109], [74, 178, 109], [76, 179, 109], [77, 179, 110], [79, 180, 110], [80, 181, 110], [81, 182, 111], [83, 182, 111], [84, 183, 111], [85, 184, 111], [87, 184, 112], [88, 185, 112], [89, 186, 112], [91, 186, 113], [92, 187, 113], [94, 188, 114], [95, 188, 114], [96, 189, 114], [98, 190, 115], [99, 190, 115], [100, 191, 116], [102, 192, 116], [103, 192, 117], [105, 193, 117], [106, 194, 118], [107, 194, 118], [109, 195, 119], [110, 196, 119], [112, 196, 120], [113, 197, 121], [114, 198, 121], [116, 198, 122], [117, 199, 122], [118, 199, 123], [120, 200, 124], [121, 201, 124], [123, 201, 125], [124, 202, 126], [125, 203, 126], [127, 203, 127], [129, 204, 128], [131, 205, 130], [133, 205, 131], [135, 206, 133], [137, 207, 134], [139, 207, 135], [141, 208, 137], [143, 208, 138], [145, 209, 140], [147, 210, 141], [149, 210, 142], [151, 211, 144], [152, 212, 145], [154, 212, 147], [156, 213, 148], [158, 213, 149], [160, 214, 151], [162, 215, 152], [164, 215, 154], [166, 216, 155], [167, 217, 157], [169, 217, 158], [171, 218, 159], [173, 218, 161], [174, 219, 162], [176, 220, 164], [178, 220, 165], [180, 221, 167], [181, 221, 168], [183, 222, 170], [185, 223, 171], [186, 223, 172], [188, 224, 174], [190, 224, 175], [191, 225, 177], [193, 226, 178], [194, 226, 180], [196, 227, 181], [198, 227, 183], [199, 228, 184], [201, 229, 185], [202, 229, 187], [204, 230, 188], [205, 230, 190], [207, 231, 191], [208, 232, 193], [210, 232, 194], [211, 233, 196], [212, 233, 197], [214, 234, 199], [215, 234, 200], [217, 235, 201], [218, 236, 203], [219, 236, 204], [221, 237, 206], [222, 237, 207]],

            maxFramerate: 60,
            fpsUpdatePeriod: 250,  // ms

            legendX: 10,
            legendY: 10,
            legendWidth: 128,
            legendHeight: 16
        }
    },
    methods: {
        datapointMoved() {

        },

        colorFromScale(t) {
            const index = Math.floor(t * (this.colorScale.length - 1));
            return this.colorScale[index];
        },

        getFill(datapointIndex) {
            if (this.coloring == "averageLocalError" && this.metrics != null) {
                return this.colorFromScale(this.metrics.average_local_error[datapointIndex]);
            } else {
                return this.fills[this.datapoints[datapointIndex].label];
            }
        },

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

            if (this.canvasWidth < this.canvasHeight) {
                this.scaling = 0.5 * (this.canvasWidth) / (maxX - minX);
            } else {
                this.scaling = 0.5 * (this.canvasHeight) / (maxY - minY);
            }

            this.xTranslation = (-meanX + 0.5) * 2 * this.scaling;
            this.yTranslation = (-meanY + 0.5) * 2 * this.scaling;

            this.selectedPointIndex = null;
            this.hoveredPointIndex = null;

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

            this.$nextTick(() => {
                this.needRerender = true;
            });
        },

        changeTransformation(p5) {
            p5.scale(this.scaling);
            p5.translate(this.xTranslation, this.yTranslation);
        },

        dataToScreenPosition(position) {
            return [
                position[0] * this.scaling + this.xTranslation,
                position[1] * this.scaling + this.yTranslation
            ]
        },

        screenToDataPosition(position) {
            return [
                (position[0] - this.xTranslation) / this.scaling,
                (position[1] - this.yTranslation) / this.scaling
            ]
        },

        datapointIndexAtMouse(p5) {
            for (const landmarkIndex of this.landmarkIndices) {
                const datapoint = this.datapoints[landmarkIndex];
                const screenPosition = this.dataToScreenPosition(datapoint.position);
                if (p5.dist(p5.mouseX, p5.mouseY, screenPosition[0], screenPosition[1]) <= this.landmarkSize) {
                    return landmarkIndex;
                }
            }

            if (this.quadtree == null) {
                return null;
            }

            // const mx = p5.mouseX / this.scaling - this.xTranslation;
            // const my = p5.mouseY / this.scaling - this.yTranslation;
            const mx = (p5.mouseX - this.xTranslation) / this.scaling;
            const my = (p5.mouseY - this.yTranslation) / this.scaling;
            const foundIndices = this.quadtree.lookup(mx, my);

            for (const foundIndex of foundIndices) {
                const datapoint = this.datapoints[foundIndex];
                const screenPosition = this.dataToScreenPosition(datapoint.position);
                if (p5.dist(p5.mouseX, p5.mouseY, screenPosition[0], screenPosition[1]) <= this.pointSize) {
                    return foundIndex;
                }
            }

            return null;
        },

        drawPoints(p5) {
            const landmarks = [];

            // Draw "normal" points
            for (const [i, datapoint] of this.datapoints.entries()) {
                if (i == this.selectedPointIndex) continue;
                if ([this.selectedPointIndex, this.hoveredPointIndex].includes(i)) continue;
                if (this.knnIndices.includes(i)) continue

                let size, shape, stroke;
                if (datapoint.is_landmark) {
                    landmarks.push({
                        position: [datapoint.position[0], datapoint.position[1]],
                        fill: this.getFill(i),
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
                    size, this.getFill(i), stroke, shape
                );
            }

            // Draw neighbor connections
            if (this.showNeighbors && this.selectedPointIndex != null) {
                p5.stroke(this.neighborStroke);
                for (const neighborIndex of this.knnIndices) {
                    const datapoint = this.datapoints[neighborIndex];
                    const datapointScreenPosition = this.dataToScreenPosition(datapoint.position);
                    const selectedPointScreenPosition = this.dataToScreenPosition(this.datapoints[this.selectedPointIndex].position);
                    p5.line(
                        datapointScreenPosition[0],
                        datapointScreenPosition[1],
                        selectedPointScreenPosition[0],
                        selectedPointScreenPosition[1]
                    );
                }
            }

            // Draw neighbors
            for (let i of this.knnIndices) {
                if (i == this.selectedPointIndex) continue;
                const datapoint = this.datapoints[i];

                let size, shape, stroke;
                if (datapoint.is_landmark) {
                    landmarks.push({
                        position: [datapoint.position[0], datapoint.position[1]],
                        fill: this.getFill(i),
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
                    size, this.getFill(i), stroke, shape
                );
            }

            // Draw selected point
            if (this.selectedPointIndex != null) {
                const datapoint = this.datapoints[this.selectedPointIndex];

                const x = (this.pointIsMoving) ? this.newPosition[0] : datapoint.position[0];
                const y = (this.pointIsMoving) ? this.newPosition[1] : datapoint.position[1];

                let size, shape, stroke;
                let is_landmark = false;
                if (datapoint.is_landmark) {
                    landmarks.push({
                        position: [x, y],
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
                        x, y,
                        size, this.selectedFill, stroke, shape
                    );
                }
            }

            // Draw hovered point
            if (this.hoveredPointIndex != null  && this.hoveredPointIndex != this.selectedPointIndex) {
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
            const screenPosition = this.dataToScreenPosition([x, y]);
            if (shape == 'circle')
                p5.circle(
                    screenPosition[0],
                    screenPosition[1],
                    size
                );
            else if (shape == 'square')
                p5.square(
                    screenPosition[0],
                    screenPosition[1],
                    size
                );
        },

        drawAxes(p5) {
            p5.stroke(255);
            p5.strokeWeight(1);
            // x axis
            p5.line(
                0,
                this.yTranslation,
                this.canvasWidth,
                this.yTranslation
            );
            // y axis
            p5.line(
                this.xTranslation,
                0,
                this.xTranslation,
                this.canvasHeight
            );
        },

        drawLegend(p5) {
            p5.push()
            p5.noStroke();

            switch(this.coloring) {
                case "label":
                    p5.fill(this.fills[0]);
                    p5.rect(this.legendX, this.legendY, this.legendHeight);
                    p5.fill(this.fills[1]);
                    p5.rect(this.legendX, this.legendY + this.legendHeight, this.legendHeight);

                    p5.fill(255);
                    p5.text("0", this.legendX + this.legendHeight + 4, this.legendY + 4);
                    p5.text("1", this.legendX + this.legendHeight + 4, this.legendY + this.legendHeight + 4);
                    break;

                case "averageLocalError":
                    for (let i = 0; i < this.legendWidth; ++i) {
                        const t = p5.map(i, 0, this.legendWidth, 0, 1);
                        const c = this.getColorFromColorScale(p5, t);
                        p5.fill(c);
                        p5.rect(i + this.legendX, this.legendY, 1, this.legendHeight);
                    }

                    p5.textAlign(p5.CENTER);
                    p5.text("0", this.legendX, this.legendY + this.legendHeight + 4);
                    p5.text("1", this.legendX + this.legendWidth, this.legendY + this.legendHeight + 4);
                    break;
            }

            p5.pop()
        },

        getColorFromColorScale(p5, t) {
            const i0 = Math.floor(t * (this.colorScale.length - 1));
            const i1 = Math.ceil(t * (this.colorScale.length - 1));
            const c0 = p5.color(this.colorScale[i0]);
            const c1 = p5.color(this.colorScale[i1]);
            return p5.lerpColor(c0, c1, t % 1);
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

        mouseInsideCanvas(p5) {
            return (
                p5.mouseX >= 0 && p5.mouseX <= this.canvasWidth &&
                p5.mouseY >= 0 && p5.mouseY <= this.canvasHeight
            );
        },

        rerender() {
            this.needRerender = true;
        },

        // P5 FUNCTIONS
        setup(p5) {
            const canvas = p5.createCanvas(this.canvasWidth, this.canvasHeight);
            canvas.elt.addEventListener('wheel', (event) => {
                event.preventDefault();
            });
            p5.frameRate(this.maxFramerate);
            p5.noStroke();
            p5.rectMode(p5.CENTER);
            this.needRerender = true;
        },
        draw(p5) {
            this.timeSinceLastFpsUpdate += p5.deltaTime;
            if (this.timeSinceLastFpsUpdate >= this.fpsUpdatePeriod) {
                while (this.timeSinceLastFpsUpdate >= this.fpsUpdatePeriod) {
                    this.timeSinceLastFpsUpdate -= this.fpsUpdatePeriod;
                }
                this.$emit('framerateChanged', p5.frameRate().toFixed(2));
            }
            if (this.needRerender) {
                p5.background(0);
                this.drawAxes(p5);
                this.drawPoints(p5);
                this.drawLegend(p5);
                this.needRerender = false;
            }
        },
        mouseDragged(p5, event) {
            if (!this.mouseInsideCanvas(p5)) {
                this.mouseReleased(p5);
                return;
            }
            if (this.selectedPointIndex == null || !this.datapoints[this.selectedPointIndex].is_landmark) {
                this.xTranslation += event.movementX;
                this.yTranslation += event.movementY;
                this.changeTransformation(p5);
            } else {
                this.newPosition = this.screenToDataPosition([p5.mouseX, p5.mouseY]);
                this.knnIndices = [];
                this.pointIsMoving = true;
            }
            this.needRerender = true;
        },
        mouseMoved(p5, event) {
            if (!this.mouseInsideCanvas(p5)) return;
            const oldHoveredPointIndex = this.hoveredPointIndex;
            this.hoveredPointIndex = this.datapointIndexAtMouse(p5);
            this.$emit('hoveredPointIndexChanged', this.hoveredPointIndex);
            if (oldHoveredPointIndex != this.hoveredPointIndex) {
                this.needRerender = true;
            }
        },
        mouseWheel(p5, event) {
            if (!this.mouseInsideCanvas(p5)) return;
            const zoom = -event.delta * this.zoomSpeed;
            this.scaling += zoom;
            this.changeTransformation(p5);
            this.needRerender = true;
        },
        mousePressed(p5) {
            if (!this.mouseInsideCanvas(p5)) return;
            this.pointIsMoving = false;
            this.selectedPointIndex = this.datapointIndexAtMouse(p5);
            this.$emit('selectedPointIndexChanged', this.selectedPointIndex);
            if (this.selectedPointIndex == null) {
                this.knnIndices = [];
                this.needRerender = true;
                return;
            }
            const datapoint = this.datapoints[this.selectedPointIndex];
            this.knnIndices = this.findKNearestNeighbors(datapoint);
            this.needRerender = true;
        },
        mouseReleased(p5) {
            if (!this.pointIsMoving) return;
            this.$emit('selectedPointMoved', this.newPosition);
            this.$nextTick(() => {
                this.knnIndices = this.findKNearestNeighbors(this.datapoints[this.selectedPointIndex]);
                this.pointIsMoving = false;
                this.newPosition = null;
            });
            this.needRerender = true;
        }
    },
    mounted() {
        const p5Script = p5 => {
            p5.setup = () => { this.p5 = p5; this.setup(p5); };
            p5.draw = () => { this.draw(p5); };
            p5.mousePressed = () => { this.mousePressed(p5); };
            p5.mouseReleased = () => { this.mouseReleased(p5); };
            p5.mouseDragged = (event) => { this.mouseDragged(p5, event); };
            p5.mouseMoved = (event) => { this.mouseMoved(p5, event); }
            p5.mouseWheel = (event) => { this.mouseWheel(p5, event); };
        }
        this.p5canvas = new P5(p5Script, 'canvas');
    }
}
</script>
