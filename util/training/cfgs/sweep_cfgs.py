sweep_cfgs = {
    "base_sweep": {
        # TODO: Add all sweepable parameters here
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-5, "max": 1e-3},
            "beta1": {"min": 0.8, "max": 0.99},
            "beta2": {"min": 0.9, "max": 0.999},
        },
    }
}