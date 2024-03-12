<style>
.prevMetricsValue0 {
  color: #222222
}
.prevMetricsValue1 {
  color: #444444
}
.prevMetricsValue2 {
  color: #666666
}
.prevMetricsValue3 {
  color: #888888
}
.prevMetricsValue4 {
  color: #aaaaaa
}
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
          :distance-metric="distanceMetric"
          :metrics="metrics"
          :labels="labels"
          @hovered-point-index-changed="hoveredPointIndexChanged"
          @selected-point-index-changed="selectedPointIndexChanged"
          @selected-point-moved="selectedPointMoved"
        />
        <div>
          <b style="color: white;">&nbsp;</b>
          <b v-if="busy" style="color: #ff0000;">    computing datapoints...</b>
          <b v-if="calculatingMetrics" style="color: #ff0000;">    calculating metrics...</b>
        </div>
      </td>
      <td style="vertical-align:top">
        <b>1. Create a new LMDS instance.</b>
        <div>
          <label for="datasetName">Dataset: </label>
          <select v-model="newDatasetName" name="datasetName">
            <option v-for="datasetName in datasetNames" :value="datasetName">{{ datasetName }}</option>
          </select>
          <br>

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
          </select>

          <button @click="updateLandmarks()" :disabled="selectedLmdsId == null || busy">Calculate</button>
        </div>
        <br>

        <b>5. Look at the metrics.</b>
        <div>
          <label for="k">k: </label>
          <input
            v-model="k" type="number" name="k" min="1" max="1000" step="1" @change="getMetrics()"
            :disabled="selectedLmds == null || !selectedLmds.points_calculated || metrics == null"
          >
          <button @click="clearHistory()" :disabled="prevMetrics.length == 0">Clear History</button>
          <br>
          <table>
            <tr>
              <th>Metric</th>
              <th>Values</th>
              <th>History →</th>
              <th v-for="_ in (prevMetricsMaxLength - 1)">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
              <th>Range</th>
            </tr>
            <tr>
              <td>Trustworthiness</td>
              <td><a v-if="metrics !== null">{{ metrics.trustworthiness.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td v-for="i in prevMetricsMaxLength" :class="'prevMetricsValue' + i">
                <a v-if="prevMetrics.length >= i">{{ prevMetrics[prevMetrics.length - i].trustworthiness.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a>
              </td>
              <td>[0 .. <b>1</b>]</td>
            </tr>
            <tr>
              <td>Continuity</td>
              <td><a v-if="metrics !== null">{{ metrics.continuity.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td v-for="i in prevMetricsMaxLength" :class="'prevMetricsValue' + i">
                <a v-if="prevMetrics.length >= i">{{ prevMetrics[prevMetrics.length - i].continuity.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a>
              </td>
              <td>[0 .. <b>1</b>]</td>
            </tr>
            <tr>
              <td>{{ this.k }}-neighborhood hit</td>
              <td><a v-if="metrics !== null">{{ metrics.neighborhood_hit.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td v-for="i in prevMetricsMaxLength" :class="'prevMetricsValue' + i">
                <a v-if="prevMetrics.length >= i">{{ prevMetrics[prevMetrics.length - i].neighborhood_hit.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a>
              </td>
              <td>[0 .. <b>1</b>]</td>
            </tr>
            <tr>
              <td>Normalized Stress</td>
              <td><a v-if="metrics !== null">{{ metrics.normalized_stress.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a></td>
              <td v-for="i in prevMetricsMaxLength" :class="'prevMetricsValue' + i">
                <a v-if="prevMetrics.length >= i">{{ prevMetrics[prevMetrics.length - i].normalized_stress.toFixed(metricsDecimalPlaces) }}</a><a v-else>-</a>
              </td>
              <td>[<b>0</b> .. 1]</td>
            </tr>
          </table>
        </div>
        <br>
        <div>
          <label for="coloring">Coloring: </label>
          <select
            v-model="coloring" name="coloring" @change="updateCanvas()"
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
      <th style="text-align: left;"><b>Hovered Point</b><a>&nbsp;</a><a style="color: #808080; background-color: #000000;">&nbsp;⬤&nbsp;</a></th>
      <th style="text-align: left;"><b>Selected Point</b><a>&nbsp;</a><a style="color: #ffffff; background-color: #000000;">&nbsp;⬤&nbsp;</a></th>
    </tr>
    <tr>
      <td style="width: 50%; vertical-align: top; text-align: left;">
        id: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].id }}</a><br>
        position: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].position }}</a><br>
        label: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].label }} ({{ labels[datapoints[hoveredPointIndex].label] }})</a><br>
        is landmark: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].is_landmark }}</a><br>
        average local error: <a v-if="hoveredPointIndex !== null">{{ metrics !== null ? metrics.average_local_error[hoveredPointIndex].toFixed(metricsDecimalPlaces) : "" }}</a><br>
        text: <a v-if="hoveredPointIndex !== null">{{ datapoints[hoveredPointIndex].text }}</a><br>
      </td>
      <td style="width: 50%; vertical-align: top; text-align: left;">
        id: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].id }}</a><br>
        position: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].position }}</a><br>
        label: <a v-if="selectedPointIndex !== null">{{ datapoints[selectedPointIndex].label }} ({{ labels[datapoints[selectedPointIndex].label] }})</a><br>
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
        fetch('http://' + this.host + ':5000/constants', {cache: "no-store"})
          .then((response) => {
              return response.json();
          }).then((data) => {
            this.datasetNames = data.dataset_names;
            this.newDatasetName = this.datasetNames[0];
            this.heuristics = data.heuristics;
            this.newHeuristic = this.heuristics[0];
            this.minLandmarkAmount = data.min_landmark_amount;
            this.maxLandmarkAmount = data.max_landmark_amount;
            this.newNumLandmarks = this.minLandmarkAmount;
            this.distanceMetrics = data.distance_metrics;
            this.newDistanceMetric = this.distanceMetrics[0];

            this.imdsAlgorithms = data.imds_algorithms;
            this.selectedImdsAlgorithm = this.imdsAlgorithms[0];
          }).catch((error) => {
              console.error(error);
          });
    },

    data() {
        return {
          // #region INSTANCE CREATION

          datasetNames: [],
          newDatasetName: null,

          heuristics: [],
          newHeuristic: null,

          distanceMetrics: [],
          newDistanceMetric: null,

          minLandmarkAmount: 10,  // default value, will be updated by server
          maxLandmarkAmount: 100,  // default value, will be updated by server
          newNumLandmarks: 10,

          seed: 42,

          // #endregion

          // #region INSTANCE SELECTION

          lmdsIds: [],
          selectedLmdsId: null,
          selectedLmds: null,

          datapoints: [],

          // #endregion

          // #region LANDMARK MOVEMENT

          hoveredPointIndex: null,
          selectedPointIndex: null,

          copiedLandmarks: {},
          initialLandmarks: {},

          // #endregion

          // #region DIMENSIONALITY REDUCTION

          imdsAlgorithms: [],
          selectedImdsAlgorithm: null,
          computingDatapoints: false,

          // #endregion

          // #region METRICS

          metrics: null,
          prevMetrics: [],
          k: 7,
          coloring: 'label',
          calculatingMetrics: false,

          // constants:
          prevMetricsMaxLength: 4,
          metricsDecimalPlaces: 3

          // #endregion
        };
    },

    methods: {
      // #region INSTANCE CREATION

      newLmds() {
        this.computingDatapoints = true;
        fetch('http://' + this.host + ':5000/lmds', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                heuristic: this.newHeuristic,
                distance_metric: this.newDistanceMetric,
                num_landmarks: parseInt(this.newNumLandmarks, 10),
                seed: parseInt(this.seed, 10),
                dataset_name: this.newDatasetName
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
            this.computingDatapoints = false;
        });
      },

      // #endregion

      // #region INSTANCE SELECTION

      lmdsSelectionChanged() {
        this.computingDatapoints = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId, {cache: "no-store"})
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
                this.computingDatapoints = false;
            });
      },

      deleteLmds() {
        this.computingDatapoints = true;
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
            this.computingDatapoints = false;
        });
      },

      // #endregion

      // #region LANDMARKS

      getLandmarks() {
        this.computingDatapoints = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/landmarks', {cache: "no-store"})
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.datapoints = data.landmarks;
                this.initialLandmarks[this.selectedLmdsId] = data.landmarks.map((landmark) => ({...landmark}));
                this.updateCanvas();
            }).catch((error) => {
                console.error(error);
            }).finally(() => {
                this.computingDatapoints = false;
            });
      },

      // #endregion

      // #region LANDMARK MOVEMENT

      copyLandmarks() {
        this.copiedLandmarks[this.selectedLmdsId] = this.datapoints.filter((landmark) => landmark.is_landmark).map((landmark) => ({...landmark}));
      },

      pasteLandmarks() {
        this.replaceLandmarks(this.copiedLandmarks[this.selectedLmdsId])
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
        this.updateCanvas();
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
        this.datapoints[this.selectedPointIndex] = datapoint;
      },

      // #endregion

      // #region DIMENSIONALITY REDUCTION

      updateLandmarks() {
        this.computingDatapoints = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/landmarks', {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                landmarks: this.datapoints.filter((landmark) => landmark.is_landmark)
            })
        }).then((response) => {
            return response.json();
        }).then((data) => {
            this.getDatapoints();
        }).catch((error) => {
            console.error(error);
            this.computingDatapoints = false;
        });
      },

      getDatapoints() {
        this.computingDatapoints = true;
        fetch('http://' + this.host + ':5000/lmds/' + this.selectedLmdsId + '/datapoints?imds_algorithm=' + this.selectedImdsAlgorithm, {cache: "no-store"})
            .then((response) => {
                return response.json();
            }).then((data) => {
                this.datapoints = data.datapoints;
                this.selectedLmdsId = data.lmds.id;
                this.selectedLmds = data.lmds;

                for (let i = this.datapoints.length - 1; i > 0; i--) {
                  const j = Math.floor(Math.random() * (i + 1));
                  [this.datapoints[i], this.datapoints[j]] = [this.datapoints[j], this.datapoints[i]];
                }

                this.updateCanvas();
                this.getMetrics();
            }).catch((error) => {
                console.error(error);
            }).finally(() => {
                this.computingDatapoints = false;
            });
      },

      // #endregion

      // #region METRICS

      clearHistory() {
        this.prevMetrics = [];
      },

      getMetrics() {
        this.calculatingMetrics = true;
        this.coloring = 'label';
        fetch(`http://${this.host}:5000/lmds/${this.selectedLmdsId}/metrics?k=${this.k}`, {cache: "no-store"})
            .then((response) => {
                return response.json();
            }).then((data) => {
                if (this.metrics !== null) {
                    this.prevMetrics = [this.metrics].concat(this.prevMetrics);
                    if (this.prevMetrics.length > this.prevMetricsMaxLength) {
                      this.prevMetrics.pop();
                    }
                }
                this.metrics = data.metrics;
                this.calculatingMetrics = false;
            }).catch((error) => {
                console.error(error);
                this.calculatingMetrics = false;
            });
      },

      // #endregion

      // #region UPDATES

      updateCanvas() {
        nextTick(() => { this.$refs.canvas.datapointsUpdated(); });
      },

      // #endregion
    },
    computed: {
        host() { return window.location.origin.split("/")[2].split(":")[0]; },
        frontendPort() { return window.location.origin.split("/")[2].split(":")[1]; },

        distanceMetric() {
          if (this.selectedLmds == null) return null;
          return this.selectedLmds.distance_metric;
        },

        landmarksPastable() {
          return this.copiedLandmarks[this.selectedLmdsId] !== undefined;
        },

        labels() {
          if (this.selectedLmds == null) return [];
          return this.selectedLmds.labels;
        }
    }
}
</script>
