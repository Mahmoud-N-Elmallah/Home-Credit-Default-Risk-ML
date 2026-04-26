from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


SAMPLERS = {
    "smote": SMOTE,
    "borderline_smote": BorderlineSMOTE,
    "adasyn": ADASYN,
    "random_undersample": RandomUnderSampler,
}

MODELS = {
    "catboost": CatBoostClassifier,
    "lightgbm": LGBMClassifier,
    "xgboost": XGBClassifier,
}

ACCELERATOR_CACHE = {}


def get_acceleration_config(config):
    return config["training"].get("acceleration", {})


def get_accelerator_order(config):
    acceleration_config = get_acceleration_config(config)
    preferred = acceleration_config.get("preferred", "cpu")
    fallback = acceleration_config.get("fallback", "cpu")
    retry = acceleration_config.get("retry_on_failure", True)
    if retry and fallback != preferred:
        return [preferred, fallback]
    return [preferred]


def get_accelerator_params(model_name, config, accelerator):
    acceleration_config = get_acceleration_config(config)
    model_config = acceleration_config.get("models", {}).get(model_name, {})
    return model_config.get(f"{accelerator}_params", {}).copy()


def accelerator_failure_is_retryable(error, config):
    message = str(error).lower()
    keywords = get_acceleration_config(config).get("retry_failure_keywords", [])
    return any(keyword.lower() in message for keyword in keywords)


def get_imbalance_config(config):
    return config["training"]["preprocessing"]["imbalance"]


def get_imbalance_sampler(config):
    imbalance_config = get_imbalance_config(config)
    strategy = imbalance_config["strategy"]
    if strategy in [None, "none", "class_weight"]:
        return None
    try:
        sampler_cls = SAMPLERS[strategy]
    except KeyError as exc:
        raise ValueError(f"Unknown imbalance strategy in config: {strategy}") from exc
    params = imbalance_config.get("sampler_params", {}).copy()
    params.setdefault("random_state", config["globals"]["random_state"])
    return sampler_cls(**params)


def split_speed_params(params, model_name=None):
    model_params = params.copy() if params else {}
    fit_options = {
        "eval_fraction": model_params.pop("eval_fraction", None),
        "early_stopping_rounds": model_params.get("early_stopping_rounds"),
    }
    if model_name != "xgboost":
        model_params.pop("early_stopping_rounds", None)
    return model_params, fit_options


def merge_model_params(name, params, config, is_trial=True, accelerator=None):
    base_params, _ = split_speed_params(params, name)
    model_params = base_params.copy()
    if accelerator is not None:
        model_params.update(get_accelerator_params(name, config, accelerator))
    model_params.setdefault("random_state", config["globals"]["random_state"])

    imbalance_config = get_imbalance_config(config)
    if imbalance_config["strategy"] == "class_weight":
        model_params.update(imbalance_config.get("class_weight_params", {}).get(name, {}))

    if name == "catboost":
        verbosity = config["training"]["verbosity"]
        verbosity_key = "catboost_trial" if is_trial else "catboost_final"
        model_params.setdefault("verbose", verbosity.get(verbosity_key, 0))
    elif name == "lightgbm":
        model_params.setdefault("verbosity", config["training"]["verbosity"].get("lgbm", -1))

    return model_params


def fit_kwargs_for_model(name, fit_options, X, y, config):
    eval_fraction = fit_options.get("eval_fraction")
    early_stopping_rounds = fit_options.get("early_stopping_rounds")
    if not eval_fraction or not early_stopping_rounds or len(X) < 100:
        return X, y, {}

    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X,
        y,
        test_size=eval_fraction,
        stratify=y,
        random_state=config["globals"]["random_state"],
    )
    if name == "catboost":
        return X_fit, y_fit, {"eval_set": (X_eval, y_eval), "early_stopping_rounds": early_stopping_rounds}
    if name == "lightgbm":
        return X_fit, y_fit, {
            "eval_set": [(X_eval, y_eval)],
            "callbacks": [early_stopping(early_stopping_rounds, verbose=False)],
        }
    if name == "xgboost":
        return X_fit, y_fit, {"eval_set": [(X_eval, y_eval)], "verbose": False}
    return X, y, {}


def capability_sample(X, y, config):
    if len(X) <= 512:
        return X, y
    sample_size = min(len(X), 512)
    _, X_sample, _, y_sample = train_test_split(
        X,
        y,
        test_size=sample_size,
        stratify=y,
        random_state=config["globals"]["random_state"],
    )
    return X_sample, y_sample


def resolve_model_accelerator(name, params, config, X, y, is_trial):
    cache_key = name
    if cache_key in ACCELERATOR_CACHE:
        return ACCELERATOR_CACHE[cache_key]

    X_sample, y_sample = capability_sample(X, y, config)
    params_no_speed, _ = split_speed_params(params, name)
    test_params = params_no_speed.copy()
    test_params.pop("early_stopping_rounds", None)
    if name == "catboost":
        test_params["iterations"] = min(int(test_params.get("iterations", 10)), 2)
    elif name in ["lightgbm", "xgboost"]:
        test_params["n_estimators"] = min(int(test_params.get("n_estimators", 10)), 2)

    accelerators = get_accelerator_order(config)
    last_error = None
    for accelerator in accelerators:
        model_params = merge_model_params(name, test_params, config, is_trial=is_trial, accelerator=accelerator)
        try:
            MODELS[name](**model_params).fit(X_sample, y_sample)
            ACCELERATOR_CACHE[cache_key] = accelerator
            return accelerator
        except Exception as error:
            last_error = error
            if accelerator != accelerators[-1] and accelerator_failure_is_retryable(error, config):
                continue
            raise
    raise last_error


def fit_model(name, params, config, X, y, is_trial):
    accelerator = resolve_model_accelerator(name, params, config, X, y, is_trial)
    model_params = merge_model_params(name, params, config, is_trial=is_trial, accelerator=accelerator)
    _, fit_options = split_speed_params(params, name)
    X_fit, y_fit, fit_kwargs = fit_kwargs_for_model(name, fit_options, X, y, config)
    model = MODELS[name](**model_params)
    try:
        model.fit(X_fit, y_fit, **fit_kwargs)
    except Exception as error:
        if accelerator_failure_is_retryable(error, config) and accelerator != get_accelerator_order(config)[-1]:
            raise RuntimeError(f"{name} failed after cached accelerator selection. Error was: {error}") from error
        raise
    return model, accelerator
