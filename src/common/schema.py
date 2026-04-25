import re


def clean_column_names(df):
    df = df.copy()
    df.columns = [re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") for col in df.columns]
    df.columns = [re.sub(r"_+", "_", col) for col in df.columns]
    return df


def expected_preprocessor_input_columns(preprocessor):
    if getattr(preprocessor, "scaler", None) is not None and hasattr(preprocessor.scaler, "feature_names_in_"):
        feature_names = preprocessor.scaler.feature_names_in_
        return feature_names.tolist() if hasattr(feature_names, "tolist") else list(feature_names)
    if getattr(preprocessor, "selector", None) is not None and hasattr(preprocessor.selector, "feature_names_in_"):
        feature_names = preprocessor.selector.feature_names_in_
        return feature_names.tolist() if hasattr(feature_names, "tolist") else list(feature_names)
    return None
