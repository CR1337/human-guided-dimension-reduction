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
          :coloring="coloring"
          :show-neighbors="showNeighbors"
          :distance-metric="distanceMetric"
          :metrics="metrics"
          @hovered-point-index-changed="hoveredPointIndexChanged"
          @selected-point-index-changed="selectedPointIndexChanged"
          @selected-point-moved="selectedPointMoved"
          @framerate-changed="framerateChanged"
        />
        <div>{{ framerate }} fps</div>
      </td>
      <td style="vertical-align:top">
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

          <label for="doPCA">Optional PCA normalisation: </label>
          <input v-model="doPCA" type="checkbox" name="doPCA">
          <br>
          <button @click="newLmds()" :disabled="busy">New LMDS</button>
        </div>
        <br>

        <div>
          <label for="lmds">LMDS: </label>
          <select v-model="selectedLmdsId" name="lmds" @change="lmdsSelectionChanged()" :disabled="busy">
            <option v-for="lmds in lmdsIds" :value="lmds">{{ lmds }}</option>
          </select>
          <br>
          heuristic: <a v-if="selectedLmdsId !== null">{{ selectedLmds.heuristic }}</a><br>
          distance metric: <a v-if="selectedLmdsId !== null">{{ selectedLmds.distance_metric }}</a><br>
          num landmarks: <a v-if="selectedLmdsId !== null">{{ selectedLmds.num_landmarks }}</a><br>
          Optional PCA normalisation: <a v-if="selectedLmdsId !== null">{{ selectedLmds.do_pca }}</a><br>
          points calculated: <a v-if="selectedLmdsId !== null">{{ selectedLmds.points_calculated }}</a><br>
          <button @click="deleteLmds()" :disabled="selectedLmdsId == null || busy">Delete</button>
        </div>
        <br>

        <div>
          <label for="k">k: </label>
          <input
            v-model="k" type="number" name="k" min="1" max="1000" step="1" @change="kChanged"
            :disabled="selectedLmds == null || !selectedLmds.points_calculated"
          ><br>
          <button @click="calculate()" :disabled="selectedLmdsId == null || busy">Calculate</button>
        </div>
        <br>

        <div >
          trustworthiness: <a v-if="metrics !== null">{{ metrics.trustworthiness.toFixed(metricsDecimalPlaces) }}</a><br>
          continuity: <a v-if="metrics !== null">{{ metrics.continuity.toFixed(metricsDecimalPlaces) }}</a><br>
          {{ this.chosenK }}-neighborhood hit: <a v-if="metrics !== null">{{ metrics.neighborhood_hit.toFixed(metricsDecimalPlaces) }}</a><br>
          <!--  shepard goodness: <a v-if="metrics !== null">{{ metrics.shepard_goodness.toFixed(metricsDecimalPlaces) }}</a><br> -->
          normalized stress: <a v-if="metrics !== null">{{ metrics.normalized_stress.toFixed(metricsDecimalPlaces) }}</a><br>
        </div>
        <div>
          <label for="coloring">Coloring: </label>
          <select
            v-model="coloring" name="coloring" @change="rerender()"
            :disabled="selectedLmds == null || !selectedLmds.points_calculated || metrics == null"
          >
            <option value="label">label</option>
            <option value="averageLocalError">average local error</option>
          </select>
        </div>
        <br>

        <div v-if="busy">
          <b style="color: #ff0000">busy...</b>
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
    position: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].position }}</a><br>
    label: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].label }}</a><br>
    is landmark: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].is_landmark }}</a><br>
    average local error: <a v-if="hoveredPointIndex !== null">{{ metrics !== null ? metrics.average_local_error[hoveredPointIndex].toFixed(metricsDecimalPlaces) : "" }}</a><br>
    text: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].text }}</a><br>
  </div>
  <br>

  <div>
    <b>Selected Point</b><a style="color: #ff0000"> ⬤</a><br>
    id: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].id }}</a><br>
    position: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].position }}</a><br>
    label: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].label }}</a><br>
    is landmark: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].is_landmark }}</a><br>
    average local error: <a v-if="selectedPointIndex !== null">{{ metrics !== null ? metrics.average_local_error[selectedPointIndex].toFixed(metricsDecimalPlaces) : "" }}</a><br>
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
          doPCA: false,

          datapoints: [],

          lmdsIds: [],
          selectedLmdsId: null,
          selectedLmds: null,
          metrics: null,

          hoveredPointIndex: null,
          selectedPointIndex: null,

          k: 7,
          chosenK: 7,
          coloring: 'label',

          busy: false,
          metricsDecimalPlaces: 3
        };
    },
    methods: {
      newLmds() {
        this.busy = true;
        fetch('http://' + this.host + ':5000/lmds', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                heuristic: this.newHeuristic,
                distance_metric: this.newDistanceMetric,
                num_landmarks: this.newNumLandmarks,
                do_pca: this.doPCA
            })
        }).then((response) => {
            return response.json();
        }).then((data) => {
            this.lmdsIds.push(data.lmds.id);
            this.selectedLmdsId = data.lmds.id;
            this.selectedLmds = data.lmds;
            this.metrics = null;
            this.getLandmarks();
        }).catch((error) => {
            console.error(error);
            this.busy = false;
        });
      },

      getLandmarks() {
        this.busy = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/landmarks')
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.datapoints = data.landmarks;
                this.updateCanvas();
            }).catch((error) => {
                console.error(error);
            }).finally(() => {
                this.busy = false;
            });
      },

      deleteLmds() {
        this.busy = true;
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
            this.metrics = null;
            this.updateCanvas();
        }).catch((error) => {
            console.error(error);
        }).finally(() => {
            this.busy = false;
        });
      },

      calculate() {
        this.updateLandmarks();
      },

      updateLandmarks() {
        this.busy = true;
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
            this.busy = false;
        });
      },

      getDatapoints() {
        this.busy = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/datapoints')
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.datapoints = data.datapoints;
                this.selectedLmdsId = data.lmds.id;
                this.selectedLmds = data.lmds;
                if (this.selectedLmds.distance_metric == 'cosine') {
                    this.calculateDatapointAngles();
                    this.sortDatapointsByAngle();
                } else {
                  // shuffle datapoints
                  for (let i = this.datapoints.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [this.datapoints[i], this.datapoints[j]] = [this.datapoints[j], this.datapoints[i]];
                  }
                }
                this.updateCanvas();
                this.getMetrics();
            }).catch((error) => {
                console.error(error);
            }).finally(() => {
                this.busy = false;
            });
      },

      kChanged() {
        this.getMetrics();
      },

      getMetrics() {
        this.metrics = null;
        this.chosenK = this.k;
        fetch(`http://${this.host}:5000/lmds/${this.selectedLmdsId}/metrics?k=${this.k}`)
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.metrics = data.metrics;
            }).catch((error) => {
                console.error(error);
            });
      },

      datapointAngle(datapoint) {
        return Math.atan2(datapoint.position[1], datapoint.position[0]);
      },

      calculateDatapointAngles() {
        for (const datapoint of this.datapoints) {
          datapoint.angle = this.datapointAngle(datapoint);
        }
      },

      sortDatapointsByAngle() {
        this.datapoints.sort((a, b) => {
            return b.angle - a.angle;
        });
      },

      updateCanvas() {
        nextTick(() => { this.$refs.canvas.datapointsUpdated(); });
      },

      rerender() {
        nextTick(() => { this.$refs.canvas.rerender(); });
      },

      lmdsSelectionChanged() {
        this.busy = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId)
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.selectedLmds = data.lmds;
                if (this.selectedLmds.points_calculated) {
                    this.getDatapoints();
                    this.getMetrics();
                } else {
                    this.getLandmarks();
                }
            }).catch((error) => {
                console.error(error);
                this.busy = false;
            });
      },

      hoveredPointIndexChanged(index) {
        this.hoveredPointIndex = index;
      },

      selectedPointIndexChanged(index) {
        this.selectedPointIndex = index;
      },

      selectedPointMoved(newPosition) {
        const datapoint = this.datapoints[this.selectedPointIndex];
        datapoint.position = newPosition;
        if (this.selectedLmds.distance_metric == 'cosine') {
            this.datapoint.angle = this.datapointAngle(datapoint);
            this.sortDatapointsByAngle();
        }
      },

      framerateChanged(framerate) {
        this.framerate = framerate;
      }
    },
    computed: {
        host() { return window.location.origin.split("/")[2].split(":")[0]; },
        frontendPort() { return window.location.origin.split("/")[2].split(":")[1]; },

        showNeighbors() {
          if (this.selectedLmds == null) return false;
          return this.selectedLmds.points_calculated;
        },

        distanceMetric() {
          if (this.selectedLmds == null) return null;
          return this.selectedLmds.distance_metric;
        }
    }
}
</script>
