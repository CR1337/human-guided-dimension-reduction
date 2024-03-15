from flask import Flask, request
from flask_cors import CORS
from dr import DimensionalityReduction
from dataset import Dataset
from idr import InverseDimensionaltyReduction
from typing import Dict, List, Any
import human_readable_ids
import pandas as pd


DEFAULT_K: int = 7
DEFAULT_NUM_LANDMARKS: int = 10
DEFAULT_SEED: int = 42

app = Flask(__name__)
CORS(app)


instances: Dict[str, DimensionalityReduction] = {}


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
        'min_landmark_amount': (
            DimensionalityReduction.LANDMARK_AMOUNT_RANGE[0]
        ),
        'max_landmark_amount': (
            DimensionalityReduction.LANDMARK_AMOUNT_RANGE[1]
        ),
        'idr_algorithms': InverseDimensionaltyReduction.VALID_NAMES,
        'dataset_names': Dataset.VALID_NAMES
    }, 200


@app.route('/instances', methods=['GET', 'POST'])
def route_instances():
    if request.method == 'GET':
        return {
            'instances': [
                instance.to_json() | {'id': id}
                for id, instance in instances.items()
            ]
        }

    elif request.method == 'POST':
        heuristic = request.json.get(
            'heuristic', DimensionalityReduction.HEURISTICS[0]
        )
        distance_metric = request.json.get(
            'distance_metric', DimensionalityReduction.DISTANCE_METRICS[0]
        )
        num_landmarks = request.json.get(
            'num_landmarks', DEFAULT_NUM_LANDMARKS
        )
        seed = request.json.get('seed', DEFAULT_SEED)
        dataset_name = request.json.get('dataset_name', Dataset.VALID_NAMES[0])
        instance_ids = (
            human_readable_ids.get_new_id().lower().replace(" ", "-")
        )
        instance = DimensionalityReduction(
            heuristic=heuristic,
            distance_metric=distance_metric,
            num_landmarks=num_landmarks,
            dataset=Dataset(dataset_name)
        )
        instances[instance_ids] = instance
        instance.select_landmarks(seed=seed)
        instance.reduce_landmarks()
        return {'instance': instance.to_json() | {'id': instance_ids}}, 201


@app.route('/instances/<instance_id>', methods=['GET', 'DELETE'])
def route_instance_id(instance_id: str):
    if request.method == 'GET':
        instance = instances.get(instance_id)
        if instance is None:
            return {"message": f"Unknown instance: {instance_id}"}, 404
        return {'instance': instance.to_json() | {'id': instance_id}}, 200

    elif request.method == 'DELETE':
        if instance_id not in instances:
            return {"message": f"Unknown instance: {instance_id}"}, 404
        del instances[instance_id]
        return {}, 200


@app.route('/instances/<instance_id>/landmarks', methods=['GET', 'PATCH'])
def route_landmarks(instance_id: str):
    instance = instances.get(instance_id)
    if instance is None:
        return {"message": f"Unknown instance: {instance_id}"}, 404

    if request.method == 'GET':
        return {'landmarks': dataframe_to_json(instance.landmarks)}, 200

    elif request.method == 'PATCH':
        new_landmarks = pd.DataFrame(request.json['landmarks']).set_index('id')
        landmarks = instance.landmarks
        landmarks.update(new_landmarks[['position', 'label']])
        instance.landmarks = landmarks
        return {}, 200


@app.route('/instances/<instance_id>/datapoints', methods=['GET'])
def route_datapoints(instance_id: str):
    instance = instances.get(instance_id)
    if instance is None:
        return {"message": f"Unknown instance: {instance_id}"}, 404

    if not instance.landmarks_reduced:
        return {"message": "Landmarks have not been reduced yet"}, 400

    idr_algorithm = request.args.get(
        'idr_algorithm', InverseDimensionaltyReduction.VALID_NAMES[0]
    )
    instance.calculate(idr_algorithm)
    return {
        'datapoints': dataframe_to_json(instance.all_points),
        'instance': instance.to_json() | {'id': instance_id}
    }, 200


@app.route('/instances/<instance_id>/metrics', methods=['GET'])
def route_instance_metrics(instance_id: str):
    instance = instances.get(instance_id)
    if instance is None:
        return {"message": f"Unknown instance: {instance_id}"}, 404

    if not instance.landmarks_reduced:
        return {"message": "Landmarks have not been reduced yet"}, 400

    k = request.args.get('k', DEFAULT_K, int)
    return {
        'metrics': instance.compute_metrics(k),
        'instance': instance.to_json() | {'id': instance_id}
    }, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
