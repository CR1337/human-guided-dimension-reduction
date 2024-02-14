from flask import Flask, request
from flask_cors import CORS
from lmds import Lmds
from typing import Dict, List, Any
from uuid import uuid4
import pandas as pd
import pickle


DEFAULT_K: int = 7
DEFAULT_HEURISTIC: str = 'random'
DEFAULT_DISTANCE_METRIC: str = 'euclidean'
DEFAULT_NUM_LANDMARKS: int = 10

USE_SMALL: bool = True

METRIC_NAMES: List[str] = [
    "trustworthiness",
    "continuity",
    "neighborhood_hit",
    "shepard_goodness",
    "average_local_error",
    "normalized_stress"
]  # TODO: put somewhere else


app = Flask(__name__)
CORS(app)


lmds_instances: Dict[str, Lmds] = {}

imdb_dataset: pd.DataFrame
with open('/server/data/imdb_embeddings.pkl', 'rb') as file:
    imdb_dataset = pickle.load(file)
imdb_dataset_small: pd.DataFrame
with open('/server/data/imdb_embeddings_small.pkl', 'rb') as file:
    imdb_dataset_small = pickle.load(file)


def dataframe_to_json(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "id": index,
            "text": series["text"],
            "label": series["label"],
            "is_landmark": series["landmark"],
            "position": (
                series["position"]
                if "position" in dataframe.columns
                else None
            )
        }
        for index, series in dataframe.iterrows()
    ]


@app.route('/', methods=['GET'])
def route_index():
    return {"message": "Hello, this is the backend!"}, 200


@app.route('/dataset', methods=['GET'])
def route_dataset():
    if USE_SMALL:
        return {'datapoints': dataframe_to_json(imdb_dataset_small)}, 200
    else:
        return {'datapoints': dataframe_to_json(imdb_dataset)}, 200


@app.route('/heuristics', methods=['GET'])
def route_heuristics():
    return {'heuristics': Lmds.HEURISTICS}, 200


@app.route('/distance-metrics', methods=['GET'])
def route_distance_metrics():
    return {'distance_metrics': Lmds.DISTANCE_METRICS}, 200


@app.route('/imds-algorithms', methods=['GET'])
def route_imds_algorithms():
    return {'imds_algorithms': Lmds.IMDS_ALGORITHMS}, 200


@app.route('/metrics', methods=['GET'])
def route_metrics():
    return {'metrics': METRIC_NAMES}, 200


@app.route('/lmds', methods=['GET', 'POST'])
def route_lmds():
    if request.method == 'GET':
        return {
            'lmds': [
                lmds.to_json() | {'id': id}
                for id, lmds in lmds_instances.items()
            ]
        }

    elif request.method == 'POST':
        heuristic = request.json.get('heuristic', DEFAULT_HEURISTIC)
        distance_metric = request.json.get(
            'distance_metric', DEFAULT_DISTANCE_METRIC
        )
        num_landmarks = request.json.get(
            'num_landmarks', DEFAULT_NUM_LANDMARKS
        )
        lmds_id = str(uuid4())
        try:
            lmds = Lmds(
                heuristic=heuristic,
                distance_metric=distance_metric,
                num_landmarks=num_landmarks,
                dataset=imdb_dataset_small if USE_SMALL else imdb_dataset,
                use_small=USE_SMALL
            )
        except NotImplementedError as ex:
            return {"message": str(ex)}, 501
        lmds_instances[lmds_id] = lmds
        lmds.select_landmarks()
        lmds.reduce_landmarks()
        return {'lmds': lmds.to_json() | {'id': lmds_id}}, 201


@app.route('/lmds/<lmds_id>', methods=['GET', 'DELETE'])
def route_lmds_lmds_id(lmds_id: str):
    if request.method == 'GET':
        lmds = lmds_instances.get(lmds_id)
        if lmds is None:
            return {"message": f"Unknown LMDS instance: {lmds_id}"}, 404
        return {'lmds': lmds.to_json() | {'id': lmds_id}}, 200

    elif request.method == 'DELETE':
        if lmds_id not in lmds_instances:
            return {"message": f"Unknown LMDS instance: {lmds_id}"}, 404
        del lmds_instances[lmds_id]
        return {}, 200


@app.route('/lmds/<lmds_id>/landmarks', methods=['GET', 'PATCH'])
def route_landmarks(lmds_id: str):
    lmds = lmds_instances.get(lmds_id)
    if lmds is None:
        return {"message": f"Unknown LMDS instance: {lmds_id}"}, 404

    if request.method == 'GET':
        return {'landmarks': dataframe_to_json(lmds.landmarks)}, 200

    elif request.method == 'PATCH':
        new_landmarks = pd.DataFrame(request.json['landmarks']).set_index('id')
        landmarks = lmds.landmarks
        landmarks.update(new_landmarks[['position', 'label']])
        lmds.landmarks = landmarks
        return {}, 200


@app.route('/lmds/<lmds_id>/datapoints', methods=['GET'])
def route_datapoints(lmds_id: str):
    lmds = lmds_instances.get(lmds_id)
    if lmds is None:
        return {"message": f"Unknown LMDS instance: {lmds_id}"}, 404
    if not lmds.landmarks_reduced:
        return {"message": "Landmarks have not been reduced yet"}, 400
    imds_algorithm = request.args.get(
        'imds_algorithm', Lmds.IMDS_ALGORITHMS[0]
    )
    do_pca = request.args.get('do_pca', "false") == "true"
    lmds.calculate(imds_algorithm, do_pca)
    return {
        'datapoints': dataframe_to_json(lmds.all_points),
        'lmds': lmds.to_json() | {'id': lmds_id}
    }, 200


@app.route('/lmds/<lmds_id>/metrics', methods=['GET'])
def route_lmds_metrics(lmds_id: str):
    lmds = lmds_instances.get(lmds_id)
    if lmds is None:
        return {"message": f"Unknown LMDS instance: {lmds_id}"}, 404
    if not lmds.landmarks_reduced:
        return {"message": "Landmarks have not been reduced yet"}, 400
    k = request.args.get('k', DEFAULT_K, int)
    return {
        'metrics': lmds.compute_metrics(k),
        'lmds': lmds.to_json() | {'id': lmds_id}
    }, 200


@app.route('/lmds/<lmds_id>/metrics/<metric_name>', methods=['GET'])
def route_lmds_metric(lmds_id: str, metric_name: str):
    lmds = lmds_instances.get(lmds_id)
    if lmds is None:
        return {"message": f"Unknown LMDS instance: {lmds_id}"}, 404
    if not lmds.landmarks_reduced:
        return {"message": "Landmarks have not been reduced yet"}, 400
    k = request.args.get('k', DEFAULT_K, int)
    if metric_name not in METRIC_NAMES:
        return {"message": f"Unknown metric: {metric_name}"}, 400
    return {
        'metric': lmds.compute_metrics(k)[metric_name],
        'lmds': lmds.to_json() | {'id': lmds_id}
    }, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
