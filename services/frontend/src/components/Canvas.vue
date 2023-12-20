<style>

</style>

<template>
    <div id="canvas"></div>
</template>

<script>
import P5 from 'p5';
import { Quadtree } from '@/util/quadtree.js'

export default {
    name: 'Canvas',
    props: ['datapoints'],
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

            landmarkFill: [255, 255, 0],
            hoveredLandmarkFill: [0, 0, 255],
            selectedLandmarkFill: [255, 0, 0],
            movingLandmarkFill: [0, 255, 0],
            pointFill: [255, 255, 255],
            hoveredPointFill: [0, 0, 255],
            selectedPointFill: [255, 0, 0],

            k: 5

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
                const x = datapoint.low_d_vector[0];
                const y = datapoint.low_d_vector[1];
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
                const x = datapoint.low_d_vector[0];
                const y = datapoint.low_d_vector[1];
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
                const px = (datapoint.low_d_vector[0] + this.xTranslation) * this.scaling;
                const py = (datapoint.low_d_vector[1] + this.yTranslation) * this.scaling;
                if (p5.dist(p5.mouseX, p5.mouseY, px, py) < this.pointSize) {
                    return foundIndex;
                }
            }

            return null;
        },



        drawPoints(p5) {
            for (const [i, datapoint] of this.datapoints.entries()) {
                this.drawPoint(
                    p5,
                    (datapoint.low_d_vector[0] + this.xTranslation) * this.scaling,
                    (datapoint.low_d_vector[1] + this.yTranslation) * this.scaling,
                    datapoint.is_landmark,
                    this.hoveredPointIndex == i,
                    this.selectedPointIndex == i,
                    false,
                    this.knnIndices.includes(i)
                );
            }
        },

        drawPoint(p5, x, y, is_landmark, is_hovered, is_selected, is_moving, is_neighbor) {
            const size = (is_landmark) ? this.landmarkSize : this.pointSize;

            let fill;
            if (is_moving)
                fill = this.movingLandmarkFill;
            else if (is_selected)
                fill = (is_landmark)
                    ? this.selectedLandmarkFill
                    : this.selectedPointFill;
            else if
                (is_hovered) fill = (is_landmark)
                    ? this.hoveredLandmarkFill
                    : this.hoveredPointFill;
            else fill = (is_landmark)
                ? this.landmarkFill
                : this.pointFill;

            if (is_neighbor)
                fill = [255, 0, 255];

            p5.fill(fill);
            (is_landmark)
                ? p5.square(x, y, size)
                : p5.circle(x, y, size);
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
            if (this.selectedPointIndex == null) return;
            const datapoint = this.datapoints[this.selectedPointIndex];
            this.knnIndices = this.quadtree.findKNearestNeighbors(
                datapoint.low_d_vector[0],
                datapoint.low_d_vector[1],
                this.k
            )
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
