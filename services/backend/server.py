from flask import Flask, request
from flask_cors import CORS
from services.backend.dr import DimensionalityReduction
from dataset import Dataset
from imds import Imds
from typing import Dict, List, Any
import human_readable_ids
import pandas as pd


DEFAULT_K: int = 7
DEFAULT_NUM_LANDMARKS: int = 10
DEFAULT_SEED: int = 42

app = Flask(__name__)
CORS(app)


lmds_instances: Dict[str, DimensionalityReduction] = {}


def dataframe_to_json(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "id": int(index),
            "text": series["text"],
            "label": str(series["label"]),
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


@app.route('/constants', methods=['GET'])
def route_constants():
    return {
        'heuristics': DimensionalityReduction.HEURISTICS,
        'distance_metrics': DimensionalityReduction.DISTANCE_METRICS,
        'min_landmark_amount': DimensionalityReduction.LANDMARK_AMOUNT_RANGE[0],
        'max_landmark_amount': DimensionalityReduction.LANDMARK_AMOUNT_RANGE[1],
        'imds_algorithms': Imds.VALID_NAMES,
        'dataset_names': Dataset.VALID_NAMES
    }, 200


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
        heuristic = request.json.get('heuristic', DimensionalityReduction.HEURISTICS[0])
        distance_metric = request.json.get(
            'distance_metric', DimensionalityReduction.DISTANCE_METRICS[0]
        )
        num_landmarks = request.json.get(
            'num_landmarks', DEFAULT_NUM_LANDMARKS
        )
        seed = request.json.get('seed', DEFAULT_SEED)
        dataset_name = request.json.get('dataset_name', Dataset.VALID_NAMES[0])
        lmds_id = human_readable_ids.get_new_id().lower().replace(" ", "-")
        lmds = DimensionalityReduction(
            heuristic=heuristic,
            distance_metric=distance_metric,
            num_landmarks=num_landmarks,
            dataset=Dataset(dataset_name)
        )
        lmds_instances[lmds_id] = lmds
        lmds.select_landmarks(seed=seed)
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
        'imds_algorithm', Imds.VALID_NAMES[0]
    )
    lmds.calculate(imds_algorithm)
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
