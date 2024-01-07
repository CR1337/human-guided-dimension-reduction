<style>

</style>

<template>
  <table>
    <tr>
      <td>
        <Canvas
          ref="canvas"
          :datapoints="datapoints"
          :k="k"
          @hovered-point-index-changed="hoveredPointIndexChanged"
          @selected-point-index-changed="selectedPointIndexChanged"
          @selected-point-moved="selectedPointMoved"
        />
      </td>
      <td>
        <div>
          <label for="heuristic">Heuristic: </label>
          <select v-model="newHeuristic" name="heuristic">
            <option v-for="heuristic in heuristics" :value="heuristic">{{ heuristic }}</option>
          </select>
          <br>

          <label for="distance-metric">Distance Metric: </label>
          <select v-model="newDistanceMetric" name="distance-metric">
            <option v-for="distanceMetric in distanceMetrics" :value="distanceMetric">{{ distanceMetric }}</option>
          </select>
          <br>

          <label for="num-landmarks">Number of Landmarks: </label>
          <input v-model="newNumLandmarks" type="number" name="num-landmarks" min="1" max="1000" step="1">
          <br>
          <button @click="newLmds()">New LMDS</button>
        </div>
        <br>

        <div>
          <label for="lmds">LMDS: </label>
          <select v-model="selectedLmdsId" name="lmds">
            <option v-for="lmds in lmdsIds" :value="lmds">{{ lmds }}</option>
          </select>
          <br>
          heuristic: <a v-if="selectedLmdsId !== null">{{ selectedLmds.heuristic }}</a><br>
          distance metric: <a v-if="selectedLmdsId !== null">{{ selectedLmds.distance_metric }}</a><br>
          num landmarks: <a v-if="selectedLmdsId !== null">{{ selectedLmds.num_landmarks }}</a><br>
          <button @click="deleteLmds()" :disabled="selectedLmdsId == null">Delete</button>
        </div>
        <br>

        <div>
          <button @click="calculate()" :disabled="selectedLmdsId == null">Calculate</button>
        </div>
        <br>

        <div>
          <label for="k">k: </label>
          <input v-model="k" type="number" name="k" min="1" max="1000" step="1">
        </div>
      </td>
    </tr>
  </table>
  <br>

  <div>
    <b>Label 0: </b><a style="color: #0000ff"> ⬤</a>
    <b>    Label 1: </b><a style="color: #ffff00"> ⬤</a>
  </div>
  <br>

  <div>
    <b>Hovered Point</b><a style="color: #00ff00"> ⬤</a><br>
    id: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].id }}</a><br>
    position: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].low_d_vector }}</a><br>
    label: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].label }}</a><br>
    is_landmark: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].is_landmark }}</a><br>
    text: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].text }}</a><br>
  </div>
  <br>

  <div>
    <b>Selected Point</b><a style="color: #ff0000"> ⬤</a><br>
    id: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].id }}</a><br>
    position: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].low_d_vector }}</a><br>
    label: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].label }}</a><br>
    is_landmark: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].is_landmark }}</a><br>
    text: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].text }}</a><br>
  </div>

</template>

<script>
import Canvas from '@/components/Canvas.vue';
import { nextTick } from 'vue';

export default {
    name: "MainPage",
    components: { Canvas },
    mounted() {
        fetch('http://' + this.host + ':5000/heuristics')
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.heuristics = data.heuristics;
                this.newHeuristic = this.heuristics[0];
            }).catch((error) => {
                console.error(error);
            });
        fetch('http://' + this.host + ':5000/distance-metrics')
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.distanceMetrics = data.distance_metrics;
                this.newDistanceMetric = this.distanceMetrics[0];
            }).catch((error) => {
                console.error(error);
            });
    },
    data() {
        return {
          heuristics: [],
          distanceMetrics: [],

          newHeuristic: null,
          newDistanceMetric: null,
          newNumLandmarks: 10,

          datapoints: [],

          lmdsIds: [],
          selectedLmdsId: null,
          selectedLmds: null,

          hoveredPointIndex: null,
          selectedPointIndex: null,

          k: 7
        };
    },
    methods: {
      newLmds() {
        fetch('http://' + this.host + ':5000/lmds', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                heuristic: this.newHeuristic,
                distance_metric: this.newDistanceMetric,
                num_landmarks: this.newNumLandmarks
            })
        }).then((response) => {
            return response.json();
        }).then((data) => {
            this.lmdsIds.push(data.lmds.id);
            this.selectedLmdsId = data.lmds.id;
            this.selectedLmds = data.lmds;
            this.getLandmarks();
        }).catch((error) => {
            console.error(error);
        });
      },

      getLandmarks() {
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/landmarks')
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.datapoints = data.landmarks;
                this.updateCanvas();
            }).catch((error) => {
                console.error(error);
            });
      },

      deleteLmds() {
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId, {
            method: 'DELETE',
        }).then((response) => {
            return response.json();
        }).then((data) => {
            this.lmdsIds = this.lmdsIds.filter((lmdsId) => lmdsId !== this.selectedLmdsId);
            this.selectedLmdsId = null;
            this.selectedLmds = null;
            this.datapoints = [];
            this.hoveredPointIndex = null;
            this.selectedPointIndex = null;
            this.updateCanvas();
        }).catch((error) => {
            console.error(error);
        });
      },

      calculate() {
        this.updateLandmarks();
      },

      updateLandmarks() {
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/landmarks', {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                landmarks: this.datapoints
            })
        }).then((response) => {
            return response.json();
        }).then((data) => {
            this.getDatapoints();
        }).catch((error) => {
            console.error(error);
        });
      },

      getDatapoints() {
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/datapoints')
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.datapoints = data.datapoints;
                this.selectedLmds = data.lmds;
                this.updateCanvas();
            }).catch((error) => {
                console.error(error);
            });
      },

      updateCanvas() {
        nextTick(() => { this.$refs.canvas.datapointsUpdated(); });
      },

      hoveredPointIndexChanged(index) {
        this.hoveredPointIndex = index;
      },

      selectedPointIndexChanged(index) {
        this.selectedPointIndex = index;
      },

      selectedPointMoved(newPosition) {
        // TODO
      }
    },
    computed: {
        host() { return window.location.origin.split("/")[2].split(":")[0]; }
    }
}
</script>
