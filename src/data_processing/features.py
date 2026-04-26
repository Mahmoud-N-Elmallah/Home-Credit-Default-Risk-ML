import logging

import numpy as np
import polars as pl

from src.data_processing.constants import (
    AGGREGATION_PREFIXES,
    APPLICATION_SOURCE_COLS,
    BASE_RATIO_FEATURES,
    DAY_UNIT,
    DAYS_EMPLOYED_COL,
    DOCUMENT_FLAG_PREFIX,
    ENQUIRY_COLS,
    EXTENDED_RATIO_FEATURES,
    EXT_SOURCES,
    EXT_SOURCE_INTERACTION_BASES,
    GLOBAL_RATIO_FEATURES,
    MISSING_INDICATOR_COLS,
    OUTPUT_FEATURES,
)
from src.data_processing.encoding import apply_frequency_encoding, encode_categoricals


logger = logging.getLogger(__name__)


def _safe_ratio_expr(numerator, denominator, output_name, eps):
    return (
        pl.col(numerator).cast(pl.Float64)
        / (pl.col(denominator).cast(pl.Float64) + pl.lit(eps))
    ).alias(output_name)


def _feature_enabled(config, feature_set):
    return feature_set in config["pipeline"]["feature_engineering"].get("enabled_sets", [])


def _available_lazy_columns(df: pl.LazyFrame):
    return set(df.collect_schema().names())


def _existing_columns(columns, available):
    return [col for col in columns if col in available]


def _mean_agg_exprs(columns):
    return [pl.col(col).mean().alias(f"{col}_mean") for col in columns]


def _sum_agg_exprs(columns):
    return [pl.col(col).sum().alias(f"{col}_sum") for col in columns]


def _max_agg_exprs(columns):
    return [pl.col(col).max().alias(f"{col}_max") for col in columns]


def _last_n_filter(df: pl.LazyFrame, group_col, order_col, n):
    rank_col = "__last_n_rank"
    return (
        df.sort([group_col, order_col], descending=[False, True])
        .with_columns(pl.col(order_col).cum_count().over(group_col).alias(rank_col))
        .filter(pl.col(rank_col) <= n)
        .drop(rank_col)
    )


def _trend_expr(recent_col, global_col, output_col):
    return (pl.col(recent_col) - pl.col(global_col)).alias(output_col)


def preprocess_base(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 2 - Preprocess base application tables."""
    logger.info("Preprocessing base tables...")
    days_birth_col = APPLICATION_SOURCE_COLS["days_birth"]
    goods_price_col = APPLICATION_SOURCE_COLS["amt_goods_price"]
    credit_col = APPLICATION_SOURCE_COLS["amt_credit"]
    days_employed_col = DAYS_EMPLOYED_COL
    anomaly_val = config["pipeline"]["anomaly_fix"]["days_employed"]
    eps = float(config["globals"]["division_epsilon"])

    def transform_base(df: pl.DataFrame):
        if days_employed_col in df.columns:
            df = df.with_columns(
                pl.when(pl.col(days_employed_col) == anomaly_val)
                .then(None)
                .otherwise(pl.col(days_employed_col))
                .alias(days_employed_col)
            )

        app_exprs = []
        if _feature_enabled(config, "application_extended"):
            if days_birth_col in df.columns:
                app_exprs.append(
                    (pl.col(days_birth_col).cast(pl.Float64).abs() / pl.lit(DAY_UNIT))
                    .alias(OUTPUT_FEATURES["age_years"])
                )
            down_payment_cols = [goods_price_col, credit_col]
            if all(col in df.columns for col in down_payment_cols):
                app_exprs.extend(
                    [
                        (
                            pl.col(goods_price_col).cast(pl.Float64)
                            - pl.col(credit_col).cast(pl.Float64)
                        ).alias(OUTPUT_FEATURES["down_payment"]),
                        (
                            (
                                pl.col(goods_price_col).cast(pl.Float64)
                                - pl.col(credit_col).cast(pl.Float64)
                            )
                            / (pl.col(credit_col).cast(pl.Float64) + pl.lit(eps))
                        ).alias(OUTPUT_FEATURES["down_payment_ratio"]),
                    ]
                )
            app_exprs.extend(
                [
                    _safe_ratio_expr(feature["numerator"], feature["denominator"], feature["name"], eps)
                    for feature in EXTENDED_RATIO_FEATURES
                    if all(col in df.columns for col in [feature["numerator"], feature["denominator"]])
                ]
            )
        if _feature_enabled(config, "missing_indicators"):
            app_exprs.extend(
                [
                    pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_is_missing")
                    for col in MISSING_INDICATOR_COLS
                    if col in df.columns and f"{col}_is_missing" not in df.columns
                ]
            )
        if app_exprs:
            df = df.with_columns(app_exprs)

        ratio_exprs = [
            _safe_ratio_expr(feature["numerator"], feature["denominator"], feature["name"], eps)
            for feature in BASE_RATIO_FEATURES
            if all(col in df.columns for col in [feature["numerator"], feature["denominator"]])
        ]
        if ratio_exprs:
            df = df.with_columns(ratio_exprs)

        doc_cols = [col for col in df.columns if col.startswith(DOCUMENT_FLAG_PREFIX)]
        if doc_cols:
            df = df.with_columns(pl.sum_horizontal(doc_cols).alias(OUTPUT_FEATURES["document_count"]))

        return df

    train = transform_base(train)
    test = transform_base(test)
    train, test = apply_frequency_encoding(train, test, config)
    train, test = encode_categoricals(train, test, config)
    return train, test


def merge_all(base: pl.DataFrame, aggs: dict, config, name=""):
    """Step 4 - Sequential merge."""
    logger.info("Merging tables for %s...", name)
    id_curr = config["training"]["id_col"]
    df = base.lazy()
    for _, agg_df in aggs.items():
        df = df.join(agg_df, on=id_curr, how="left")

    df = df.collect()
    logger.info("Shape after all merges: %s", df.shape)
    high_null_threshold = config["pipeline"]["high_null_threshold"]
    null_rates = [df.get_column(col).null_count() / df.height for col in df.columns]
    high_nulls = [
        col for col, rate in zip(df.columns, null_rates)
        if rate > high_null_threshold and col not in base.columns
    ]
    if high_nulls:
        logger.warning("%s columns with high null rate (>%s).", len(high_nulls), high_null_threshold)
    return df


def impute_missing(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 5 - Missing value imputation."""
    logger.info("Imputing missing values...")
    fill_values = config["pipeline"]["fill_values"]
    aux_prefixes = AGGREGATION_PREFIXES
    target_col = config["training"]["target_col"]
    id_col = config["training"]["id_col"]

    aux_cols = [col for col in train.columns if any(col.startswith(prefix) for prefix in aux_prefixes)]
    train = train.with_columns([pl.col(col).fill_null(fill_values["aux_missing"]) for col in aux_cols])
    test_aux_cols = [col for col in test.columns if col in aux_cols]
    if test_aux_cols:
        test = test.with_columns([pl.col(col).fill_null(fill_values["aux_missing"]) for col in test_aux_cols])

    base_cols = [
        col for col in train.columns
        if col not in aux_cols
        and col not in [target_col, id_col]
        and train.get_column(col).dtype in pl.NUMERIC_DTYPES
    ]
    medians = train.select([pl.col(col).median() for col in base_cols])

    train_exprs = []
    test_exprs = []
    for col in base_cols:
        med_val = medians.get_column(col)[0]
        if med_val is None:
            med_val = fill_values["generated_missing"]
        if train.get_column(col).null_count() > 0:
            train_exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_is_missing"))
            train_exprs.append(pl.col(col).fill_null(med_val))
            if col in test.columns:
                test_exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_is_missing"))
                test_exprs.append(pl.col(col).fill_null(med_val))
        elif col in test.columns and test.get_column(col).null_count() > 0:
            test_exprs.append(pl.col(col).fill_null(med_val))

    if train_exprs:
        train = train.with_columns(train_exprs)
    if test_exprs:
        test = test.with_columns(test_exprs)
    return train, test


def add_global_features(df: pl.DataFrame, config):
    """Step 6 - Global feature engineering."""
    logger.info("Adding global features...")
    eps = float(config["globals"]["division_epsilon"])
    fill_value = config["pipeline"]["fill_values"]["generated_missing"]

    exprs = [
        _safe_ratio_expr(feature["numerator"], feature["denominator"], feature["name"], eps)
        for feature in GLOBAL_RATIO_FEATURES
        if all(col in df.columns for col in [feature["numerator"], feature["denominator"]])
    ]

    ext_cols = EXT_SOURCES
    if all(col in df.columns for col in ext_cols):
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in ext_cols])
        exprs.append(pl.mean_horizontal(ext_cols).alias(OUTPUT_FEATURES["ext_sources_mean"]))
        exprs.append(pl.min_horizontal(ext_cols).alias(OUTPUT_FEATURES["ext_sources_min"]))
        exprs.append(pl.max_horizontal(ext_cols).alias(OUTPUT_FEATURES["ext_sources_max"]))
        exprs.append(
            (pl.max_horizontal(ext_cols) - pl.min_horizontal(ext_cols))
            .alias(OUTPUT_FEATURES["ext_sources_range"])
        )
        exprs.append(
            (pl.col(ext_cols[0]) * pl.col(ext_cols[1]) * pl.col(ext_cols[2]))
            .alias(OUTPUT_FEATURES["ext_sources_prod"])
        )
        mean = pl.mean_horizontal(ext_cols)
        exprs.append(
            pl.mean_horizontal([(pl.col(col) - mean) ** 2 for col in ext_cols])
            .sqrt()
            .alias(OUTPUT_FEATURES["ext_sources_std"])
        )
        for base_col in EXT_SOURCE_INTERACTION_BASES:
            if base_col in df.columns:
                for ext_col in ext_cols:
                    exprs.append(_safe_ratio_expr(base_col, ext_col, f"{base_col}_TO_{ext_col}", eps))

    enq_cols = ENQUIRY_COLS
    if all(col in df.columns for col in enq_cols):
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in enq_cols])
        exprs.append(pl.sum_horizontal(enq_cols).alias(OUTPUT_FEATURES["enquiry_total"]))

    if exprs:
        df = df.with_columns(exprs)

    num_cols = [col for col in df.columns if df[col].dtype in pl.NUMERIC_DTYPES]
    df = df.with_columns(
        [
            pl.when(pl.col(col).is_infinite() | pl.col(col).is_nan())
            .then(None)
            .otherwise(pl.col(col))
            .fill_null(fill_value)
            .alias(col)
            for col in num_cols
        ]
    )
    return df


def feature_cleanup(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 7 - Feature cleanup."""
    logger.info("Cleaning up features...")
    corr_threshold = config["pipeline"]["correlation_threshold"]
    var_threshold = config["pipeline"]["variance_threshold"]
    fill_value = config["pipeline"]["fill_values"]["test_missing_column"]
    target_col = config["training"]["target_col"]
    id_col = config["training"]["id_col"]

    num_cols = [
        col for col in train.columns
        if train[col].dtype in pl.NUMERIC_DTYPES and col not in [target_col, id_col]
    ]
    train_pd = train.select(num_cols).to_pandas()
    corr_matrix = train_pd.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    logger.info("Dropping %s correlated cols", len(to_drop_corr))
    train = train.drop(to_drop_corr, strict=False)

    to_drop_var = []
    for col in train.columns:
        if col in [target_col, id_col]:
            continue
        counts = train.get_column(col).value_counts()
        count_col = [name for name in counts.columns if name != col][0]
        max_prop = counts.get_column(count_col).max() / train.height
        if max_prop > var_threshold:
            to_drop_var.append(col)

    logger.info("Dropping %s low var cols", len(to_drop_var))
    train = train.drop(to_drop_var, strict=False)

    test_cols = [col for col in train.columns if col != target_col]
    missing_in_test = [col for col in test_cols if col not in test.columns]
    if missing_in_test:
        test = test.with_columns([pl.lit(fill_value).alias(col) for col in missing_in_test])
    test = test.select(test_cols)
    cleanup_info = {
        "dropped_correlated": sorted(to_drop_corr),
        "dropped_low_variance": sorted(to_drop_var),
        "missing_columns_added_to_test": sorted(missing_in_test),
    }
    return train, test, cleanup_info


def validate(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 8 - Validation."""
    logger.info("Validating...")
    errors = []
    target_col = config["training"]["target_col"]
    id_col = config["training"]["id_col"]

    if target_col in test.columns:
        errors.append(f"{target_col} in test")
    if target_col not in train.columns:
        errors.append(f"{target_col} not in train")
    if sum(test.null_count().row(0)) > 0:
        errors.append("Test has nulls")
    if sum(train.null_count().row(0)) > 0:
        errors.append("Train has nulls")
    if train.get_column(id_col).n_unique() != train.height:
        errors.append("Train ID not unique")
    if test.get_column(id_col).n_unique() != test.height:
        errors.append("Test ID not unique")
    expected_test_cols = [col for col in train.columns if col != target_col]
    if test.columns != expected_test_cols:
        errors.append("Train/test feature columns are misaligned")

    if errors:
        raise ValueError("Validation failed: " + "; ".join(errors))

    logger.info("Validation passed. Train: %s, Test: %s", train.shape, test.shape)
