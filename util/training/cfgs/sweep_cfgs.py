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
            "weight_decay": {"min": 0.0, "max": 0.1},
        },
    },
    "OneModel_detailed_sweep": {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 0.2, "max": 0.6},
            "model_param1": {"values": [8, 16, 32, 64]},
            "inner_activation": {"values": ["relu", "sigmoid"]},
            "end_activation": {"values": ["identity"]},
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
