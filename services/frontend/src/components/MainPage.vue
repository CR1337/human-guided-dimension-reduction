<style>

</style>

<template>
  Hello, World!
  <Canvas ref="canvas" :datapoints="datapoints"/>
</template>

<script>
import Canvas from '@/components/Canvas.vue';
import { nextTick } from 'vue';

export default {
    name: "MainPage",
    data() {
        return {
            datapoints: []
        };
    },
    methods: {},
    components: { Canvas },
    mounted() {
      fetch('http://localhost:5000/datapoints?amount=10&high_d_vector_size=768&low_d_vector_size=2')
        .then(response => response.json())
        .then(data => {
          this.datapoints = data;
          nextTick(() => {
            console.log(this.$refs.canvas);
            this.$refs.canvas.datapointsUpdated();
          });
        });
    }
}
</script>
