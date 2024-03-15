sweep_cfgs = {
    "OneModel_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "program": "train.py --config util/training/cfgs/train.yml",
        "parameters": {
            "learning_rate": {"min": 1e-3, "max": 1e0},
            "model_param1": {"min": 10, "max": 1_000},
            "inner_activation": {
                "values": ["relu", "tanh", "sigmoid", "identity"]
            },
            "end_activation": {"values": ["relu", "identity"]},
        },
    },
    "TwoModel_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "program": "train.py --config util/training/cfgs/train.yml",
        "parameters": {
            "learning_rate": {"min": 1e-3, "max": 1e0},
            "model_param1": {"min": 10, "max": 1_000},
            "model_param2": {"min": 10, "max": 1_000},
            "inner_activation": {
                "values": ["relu", "tanh", "sigmoid", "identity"]
            },
            "end_activation": {"values": ["relu", "identity"]},
        },
    }
}
