from flask import Flask, request
from flask_cors import CORS
from datapoint import Datapoint
from typing import Any, Dict, Tuple


app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def route_index():
    return {"message": "Hello, this is the backend!"}, 200


def read_required_query_parameter(
    key: str, type: type
) -> Tuple[Any | None, Tuple[Dict[str, str], int] | None]:
    try:
        if (value := request.args.get(key, type=type)) is None:
            return None, (
                {"message": f"Missing '{key}' query parameter!"}, 400
            )
    except ValueError:
        return None, (
            {"message": f"Invalid '{key}' query parameter!"}, 400
        )
    return value, None


def read_optional_query_parameter(
    key: str, type: type
) -> Tuple[Any | None, Tuple[Dict[str, str], int] | None]:
    try:
        if (value := request.args.get(key, type=type)) is None:
            return None, None
    except ValueError:
        return None, (
            {"message": f"Invalid '{key}' query parameter!"}, 400
        )
    return value, None


@app.route('/datapoints', methods=['GET'])
def route_datapoints():
    parameters = {}
    for key, type, func in [
        ("amount", int, read_required_query_parameter),
        ("high_d_vector_size", int, read_required_query_parameter),
        ("low_d_vector_size", int, read_required_query_parameter),
        ("labels", str, read_optional_query_parameter),
        ("landmark_ratio", float, read_optional_query_parameter),
        ("generate_random_data", bool, read_optional_query_parameter),
    ]:
        value, error = func(key, type)
        if error is not None:
            return error
        if value is not None:
            parameters[key] = value

    if parameters.get("labels") is not None:
        parameters["labels"] = parameters["labels"].split(",")

    datapoints = Datapoint.generate_random_bulk(**parameters)

    return [d.to_dict() for d in datapoints], 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
