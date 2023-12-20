<style>

</style>

<template>
  <Canvas
    ref="canvas"
    :datapoints="datapoints"
    @hovered-point-index-changed="hoveredPointIndexChanged"
    @selected-point-index-changed="selectedPointIndexChanged"
  />
  <div>{{ text }}</div>
</template>

<script>
import Canvas from '@/components/Canvas.vue';
import { nextTick } from 'vue';

export default {
    name: "MainPage",
    data() {
        return {
            datapoints: [],
            text: "Hi"
        };
    },
    methods: {
      hoveredPointIndexChanged(index) {

      },

      selectedPointIndexChanged(index) {
        if (index != null) {
          this.text = JSON.stringify(this.datapoints[index].data);
          this.text += "\n";
          this.text += JSON.stringify(this.datapoints[index].low_d_vector);
        } else {
          this.text = "";
        }
      }
    },
    components: { Canvas },
    mounted() {
      fetch('http://localhost:5000/datapoints?amount=1000&high_d_vector_size=768&low_d_vector_size=2&generate_random_data=true&landmark_ratio=0.1')
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
