<style>

</style>

<template>
    <div id="canvas"></div>
</template>

<script>
import P5 from 'p5';

export default {
    name: 'Canvas',
    props: ['datapoints'],
    data() {
        return {
            p5canvas: null,
            xTranslation: 0.0,
            yTranslation: 0.0,
            xScale: 1.0,
            yScale: 1.0,
            canvasWidth: 800,
            canvasHeight: 800,
            margin: 0.3,
            selectedPointIndex: null
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
            console.log(this.datapoints);
            for (const datapoint of this.datapoints) {
                const x = datapoint.low_d_vector[0];
                const y = datapoint.low_d_vector[1];
                console.log(x + ", " + y)
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                sumX += x;
                sumY += y;
            }
            const meanX = sumX / this.datapoints.length;
            const meanY = sumY / this.datapoints.length;
            this.xScale = (1.0 - this.margin) * ((maxX - minX) / this.canvasWidth);
            this.yScale = (1.0 - this.margin) * ((maxY - minY) / this.canvasHeight);
            this.xTranslation = (this.xScale * this.canvasWidth / 2.0) - meanX;
            this.yTranslation = (this.yScale * this.canvasHeight / 2.0) - meanY;

            console.log("sumX: " + sumX);
            console.log("sumY: " + sumY);
            console.log("meanX: " + meanX);
            console.log("meanY: " + meanY);
            console.log("minX: " + minX);
            console.log("maxX: " + maxX);
            console.log("minY: " + minY);
            console.log("maxY: " + maxY);
            console.log("xTranslation: " + this.xTranslation);
            console.log("yTranslation: " + this.yTranslation);
            console.log("xScale: " + this.xScale);
            console.log("yScale: " + this.yScale);
        },

        setup(p5) {
            p5.createCanvas(this.canvasWidth, this.canvasHeight);
        },
        draw(p5) {
            p5.background(0);
            this.drawPoints(p5);
        },
        mousePressed(p5) {
            for (const datapoint of this.datapoints) {
                const x = datapoint.low_d_vector[0];
                const y = datapoint.low_d_vector[1];
                if (p5.dist(x, y, p5.mouseX, p5.mouseY) < 10) {
                    this.selectedPointIndex = this.datapoints.indexOf(datapoint);
                    break;
                }
            }
        },
        mouseReleased(p5) {
            console.log("mouseReleased");
        },
        mouseDragged(p5, event) {
            this.xTranslation += (event.movementX * this.xScale);
            this.yTranslation += (event.movementY * this.yScale);
        },
        mouseWheel(p5, event) {
            this.xScale += event.delta / 1000000.0;
            this.yScale += event.delta / 1000000.0;
        },

        drawPoints(p5) {
            for (const [i, datapoint] of this.datapoints.entries()) {
                const x = datapoint.low_d_vector[0];
                const y = datapoint.low_d_vector[1];
                if (i == this.selectedPointIndex) {
                    p5.fill(255, 0, 0);
                } else {
                    p5.fill(255);
                }
                p5.ellipse(
                    (x + this.xTranslation) / this.xScale,
                    (y + this.yTranslation) / this.yScale,
                    10,
                    10
                );
            }
        }
    },
    mounted() {
        const p5Script = p5 => {
            p5.setup = () => { this.setup(p5); };
            p5.draw = () => { this.draw(p5); };
            p5.mousedPressed = () => { this.mousePressed(p5); };
            p5.mouseReleased = () => { this.mouseReleased(p5); };
            p5.mouseDragged = (event) => { this.mouseDragged(p5, event); };
            p5.mouseWheel = (event) => { this.mouseWheel(p5, event); };
        }
        this.p5canvas = new P5(p5Script, 'canvas');
    }
}
</script>
