from copy import deepcopy


def resolve_training_config(config):
    resolved = deepcopy(config)
    training_config = resolved["training"]
    profile_name = training_config["run_mode"]
    profiles = training_config.get("run_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Unknown training.run_mode: {profile_name}")

    profile = profiles[profile_name]
    for key in ["cv_splits", "optuna_n_trials", "optuna_subsample_rate"]:
        training_config[key] = profile[key]

    feature_selection = training_config["preprocessing"]["feature_selection"]
    feature_selection["enabled_during_search"] = profile["feature_selection_enabled_during_search"]
    training_config["run_full_oof_validation"] = profile["run_full_oof_validation"]
    return resolved


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
