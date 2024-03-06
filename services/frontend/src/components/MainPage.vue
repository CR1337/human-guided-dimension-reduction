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
        <div>
          {{ framerate }} fps
          <b v-if="busy" style="color: #ff0000;">    busy...</b>
          <b v-if="calculatingMetrics" style="color: #ff0000;">    calculating metrics...</b>
        </div>
      </td>
      <td style="vertical-align:top">
        <b>1. Create a new LMDS instance.</b>
        <div>
          <label for="heuristic">Landmark selection heuristic: </label>
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
          <input v-model="newNumLandmarks" type="range" name="num-landmarks" :min="minLandmarkAmount" :max="maxLandmarkAmount" step="1">
          <a>{{ newNumLandmarks }}</a>
          <br>

          <label for="seed">Seed: </label>
          <input v-model="seed" type="number" name="seed" min="0" step="1">
          <br>

          <button @click="newLmds()" :disabled="busy">New LMDS</button>
        </div>
        <br>

        <b>2. Select one of all created LMDS instances.</b>
        <div>
          <label for="lmds">LMDS: </label>
          <select v-model="selectedLmdsId" name="lmds" @change="lmdsSelectionChanged()" :disabled="busy">
            <option v-for="lmds in lmdsIds" :value="lmds">{{ lmds }}</option>
          </select>
          <br>
          Landmark selection heuristic: <a v-if="selectedLmdsId !== null">{{ selectedLmds.heuristic }}</a><br>
          Distance metric: <a v-if="selectedLmdsId !== null">{{ selectedLmds.distance_metric }}</a><br>
          Number of landmarks: <a v-if="selectedLmdsId !== null">{{ selectedLmds.num_landmarks }}</a><br>
          Points calculated: <a v-if="selectedLmdsId !== null">{{ selectedLmds.points_calculated }}</a><br>
          <button @click="deleteLmds()" :disabled="selectedLmdsId == null || busy">Delete</button>
        </div>
        <br>

        <b>3. Move the landmarks.</b><br>
        <div>
          <button @click="copyLandmarks()" :disabled="selectedLmdsId == null">Copy Landmarks</button>
          <button @click="pasteLandmarks()" :disabled="selectedLmdsId == null || !landmarksPastable">Paste Landmarks</button>
          <button @click="resetLandmarks()" :disabled="selectedLmdsId == null">Reset Landmarks</button>
        </div>
        <br>

        <b>4. Perform the dimensionality reduction.</b>
        <div>
          <label for="imds">Inverse MDS algorithm: </label>
          <select v-model="selectedImdsAlgorithm" name="imds" :disabled="selectedLmds == null">
            <option v-for="algorithm in imdsAlgorithms" :value="algorithm">{{ algorithm }}</option>
          </select><br>

          <label for="doPCA">PCA normalisation: </label>
          <input v-model="doPCA" type="checkbox" name="doPCA" :disabled="selectedLmds == null">
          <br>

          <button @click="calculate()" :disabled="selectedLmdsId == null || busy">Calculate</button>
        </div>
        <br>

        <b>5. Look at the metrics.</b>
        <div>
          <label for="k">k: </label>
          <input
            v-model="k" type="number" name="k" min="1" max="1000" step="1" @change="kChanged"
            :disabled="selectedLmds == null || !selectedLmds.points_calculated || metrics == null"
          ><br>
          <table>
            <tr>
              <th>Metric</th>
              <th>Value</th>
              <th>Range</th>
            </tr>
            <tr>
              <td>Trustworthiness</td>
              <td><a v-if="metrics !== null">{{ metrics.trustworthiness.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td>[0 .. <b>1</b>]</td>
            </tr>
            <tr>
              <td>Continuity</td>
              <td><a v-if="metrics !== null">{{ metrics.continuity.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td>[0 .. <b>1</b>]</td>
            </tr>
            <tr>
              <td>{{ this.k }}-neighborhood hit</td>
              <td><a v-if="metrics !== null">{{ metrics.neighborhood_hit.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td>[0 .. <b>1</b>]</td>
            </tr>
            <!--
            <tr>
              <td>Shepard Goodness</td>
              <td><a v-if="metrics !== null">{{ metrics.shepard_goodness.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td>[0 .. <b>1</b>]</td>
            </tr>
            -->
            <tr>
              <td>Normalized Stress</td>
              <td><a v-if="metrics !== null">{{ metrics.normalized_stress.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td>[<b>0</b> .. 1]</td>
            </tr>
          </table>
        </div>
        <br>
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
      </td>
    </tr>
  </table>

  <table style="width: 100%;">
    <tr>
      <th style="text-align: left;"><b>Hovered Point</b><a style="color: #00ff00"> ⬤</a></th>
      <th style="text-align: left;"><b>Selected Point</b><a style="color: #ff0000"> ⬤</a></th>
    </tr>
    <tr>
      <td style="width: 50%; vertical-align: top; text-align: left;">
        id: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].id }}</a><br>
        position: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].position }}</a><br>
        label: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].label }}</a><br>
        is landmark: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].is_landmark }}</a><br>
        average local error: <a v-if="hoveredPointIndex !== null">{{ metrics !== null ? metrics.average_local_error[hoveredPointIndex].toFixed(metricsDecimalPlaces) : "" }}</a><br>
        text: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].text }}</a><br>
      </td>
      <td style="width: 50%; vertical-align: top; text-align: left;">
        id: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].id }}</a><br>
        position: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].position }}</a><br>
        label: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].label }}</a><br>
        is landmark: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].is_landmark }}</a><br>
        average local error: <a v-if="selectedPointIndex !== null">{{ metrics !== null ? metrics.average_local_error[selectedPointIndex].toFixed(metricsDecimalPlaces) : "" }}</a><br>
        text: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].text }}</a><br>
      </td>
    </tr>
  </table>
</template>

<script>
import Canvas from '@/components/Canvas.vue';
import { nextTick } from 'vue';

export default {
    name: "MainPage",
    components: { Canvas },
    mounted() {
        fetch('http://' + this.host + ':5000/constants')
          .then((response) => {
              return response.json();
          }).then((data) => {
              this.heuristics = data.heuristics;
              this.newHeuristic = this.heuristics[0];
              this.distanceMetrics = data.distance_metrics;
              this.newDistanceMetric = this.distanceMetrics[0];
              this.minLandmarkAmount = data.min_landmark_amount;
              this.maxLandmarkAmount = data.max_landmark_amount;
              this.newNumLandmarks = this.minLandmarkAmount;
              this.imdsAlgorithms = data.imds_algorithms;
              this.selectedImdsAlgorithm = this.imdsAlgorithms[0];
          }).catch((error) => {
              console.error(error);
          });
    },
    data() {
        return {
          heuristics: [],
          distanceMetrics: [],
          minLandmarkAmount: 10,  // default value, will be updated by server
          maxLandmarkAmount: 100,  // default value, will be updated by server
          imdsAlgorithms: [],

          newHeuristic: null,
          newDistanceMetric: null,
          newNumLandmarks: 10,
          doPCA: false,
          seed: 42,

          datapoints: [],
          copiedLandmarks: {},
          initialLandmarks: {},

          lmdsIds: [],
          selectedLmdsId: null,
          selectedLmds: null,
          metrics: null,

          hoveredPointIndex: null,
          selectedPointIndex: null,

          k: 7,
          selectedImdsAlgorithm: null,
          coloring: 'label',

          busy: false,
          calculatingMetrics: false,
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
                num_landmarks: parseInt(this.newNumLandmarks, 10),
                seed: parseInt(this.seed, 10)
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
                this.initialLandmarks[this.selectedLmdsId] = data.landmarks.map((landmark) => ({...landmark}));
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

      copyLandmarks() {
        this.copiedLandmarks[this.selectedLmdsId] = this.datapoints.filter((landmark) => landmark.is_landmark).map((landmark) => ({...landmark}));
      },

      pasteLandmarks() {
        this.replaceLandmarks(this.copiedLandmarks[this.selectedLmdsId])
        // let datapoints = this.datapoints.map((datapoint) => ({...datapoint}));
        // for (const landmark of this.copiedLandmarks[this.selectedLmdsId]) {
        //   const index = this.datapoints.findIndex((datapoint) => datapoint.id === landmark.id);
        //   datapoints.splice(index, 1, landmark);
        // }
        // this.datapoints = datapoints.map((datapoint) => ({...datapoint}));

        // if (this.selectedLmds.distance_metric == 'cosine') {
        //     this.sortDatapointsByAngle();
        // }

        // this.rerender();
      },

      resetLandmarks() {
        this.replaceLandmarks(this.initialLandmarks[this.selectedLmdsId]);
      },

      replaceLandmarks(landmarks) {
        let datapoints = this.datapoints.map((datapoint) => ({...datapoint}));
        for (const landmark of landmarks) {
          const index = this.datapoints.findIndex((datapoint) => datapoint.id === landmark.id);
          datapoints.splice(index, 1, landmark);
        }
        this.datapoints = datapoints.map((datapoint) => ({...datapoint}));

        if (this.selectedLmds.distance_metric == 'cosine') {
            this.sortDatapointsByAngle();
        }

        this.rerender();
      },

      getDatapoints() {
        this.busy = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/datapoints?imds_algorithm=' + this.selectedImdsAlgorithm + "&do_pca=" + this.doPca)
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
        this.calculatingMetrics = true;
        this.coloring = 'label';
        fetch(`http://${this.host}:5000/lmds/${this.selectedLmdsId}/metrics?k=${this.k}`)
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.metrics = data.metrics;
                this.calculatingMetrics = false;
            }).catch((error) => {
                console.error(error);
                this.calculatingMetrics = false;
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
        this.datapoints[this.selectedPointIndex] = datapoint;
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
        },

        landmarksPastable() {
          return this.copiedLandmarks[this.selectedLmdsId] !== undefined;
        }
    }
}
</script>
