<style>

</style>

<template>
    <div id="canvas"></div>
</template>

<script>
import P5 from 'p5';

export default {
    name: 'Canvas',
    data() {
        return {
            p5canvas: null
        }
    },
    methods: {
        setup(p5) {
            p5.createCanvas(400, 400);
        },
        draw(p5) {
            p5.background(0);
            p5.fill(255);
            p5.rect(100, 100, 200, 200);
        },
    },
    mounted() {
        const p5Script = p5 => {
            p5.setup = () => { this.setup(p5) };
            p5.draw = () => { this.draw(p5) };
        }
        this.p5canvas = new P5(p5Script, 'canvas');
    }
}
</script>
