def primary_model_name(config):
    return config["training"]["models"]["primary"]


def get_primary_estimator_config(config):
    primary = primary_model_name(config)
    return get_estimator_config_by_name(config, primary)


def get_estimator_config_by_name(config, name):
    for candidate in config["training"]["models"]["candidates"]:
        if candidate["name"] == name:
            return candidate
    raise ValueError(f"Model not found in training.models.candidates: {name}")
