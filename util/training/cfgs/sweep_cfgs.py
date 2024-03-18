sweep_cfgs = {
    "OneModel_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-3, "max": 1e-1},
            "model_param1": {"min": 10, "max": 1_000},
            "inner_activation": {"values": ["relu", "tanh", "sigmoid", "identity"]},
            "end_activation": {"values": ["relu", "identity"]},
            "batch_size": {"values": [32, 64, 128, 256]},
            "dropout_prob": {"min": 0.1, "max": 0.5},
        },
    },
    "OneModel_detailed_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.001, "max": 0.005},
            "inner_activation": {"values": ["relu", "sigmoid", "identity"]},
            "end_activation": {"values": ["identity"]},
            "batch_size": {"values": [32, 64, 128]},
        },
    },
    "TwoModel_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-3, "max": 1e0},
            "model_param1": {"min": 10, "max": 1_000},
            "model_param2": {"min": 10, "max": 1_000},
            "inner_activation": {"values": ["relu", "tanh", "sigmoid", "identity"]},
            "end_activation": {"values": ["relu", "identity"]},
        },
    },
}
