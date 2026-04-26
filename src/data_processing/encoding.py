import logging

import polars as pl

from src.data_processing.constants import FREQUENCY_ENCODING_COLS


logger = logging.getLogger(__name__)


def get_proportions(df: pl.LazyFrame, target_col: str, prefix: str):
    """Generate value-proportion expressions from categories present in a table."""
    categories = (
        df.select(target_col)
        .drop_nulls()
        .unique()
        .collect()
        .get_column(target_col)
        .to_list()
    )
    return [
        pl.col(target_col).eq(category).mean().alias(f"{prefix}_{category}_prop")
        for category in sorted(categories, key=str)
    ]


def _sorted_unique_values(df: pl.DataFrame, col: str, include_null: bool, null_label: str):
    values = df.get_column(col).drop_nulls().unique().to_list()
    values = sorted(values, key=str)
    if include_null and df.get_column(col).null_count() > 0:
        values.append(null_label)
    return values


def _categorical_expr(col, category, alias, null_label):
    return (
        pl.col(col)
        .fill_null(null_label)
        .eq(category)
        .cast(pl.UInt8)
        .alias(alias)
    )


def _binary_expr(col, mapping, unknown_value):
    expr = None
    for category, encoded_value in mapping.items():
        branch = pl.when(pl.col(col).eq(category)).then(pl.lit(encoded_value))
        expr = branch if expr is None else expr.when(pl.col(col).eq(category)).then(pl.lit(encoded_value))
    return expr.otherwise(pl.lit(unknown_value)).cast(pl.Int8).alias(col)


def encode_categoricals(train: pl.DataFrame, test: pl.DataFrame, config):
    cat_config = config["pipeline"]["categorical_encoding"]
    target_col = config["training"]["target_col"]
    id_col = config["training"]["id_col"]
    null_label = cat_config["null_category_label"]
    drop_first = cat_config["one_hot_drop_first"]
    unknown_value = cat_config["unknown_value"]
    binary_values = cat_config["binary_values"]

    cat_cols = [
        col for col in train.columns
        if train[col].dtype == pl.String and col not in [target_col, id_col]
    ]
    binary_cols = 0
    dummy_cols = 0

    for col in cat_cols:
        train_values = _sorted_unique_values(train, col, include_null=True, null_label=null_label)
        non_null_values = [value for value in train_values if value != null_label]

        if cat_config["binary_strategy"] == "ordinal" and len(non_null_values) == 2:
            mapping = dict(zip(non_null_values, binary_values))
            train = train.with_columns(_binary_expr(col, mapping, unknown_value))
            if col in test.columns:
                test = test.with_columns(_binary_expr(col, mapping, unknown_value))
            else:
                test = test.with_columns(pl.lit(unknown_value, dtype=pl.Int8).alias(col))
            binary_cols += 1
            continue

        categories = train_values[1:] if drop_first and train_values else train_values
        train_exprs = [
            _categorical_expr(col, category, f"{col}_{category}", null_label)
            for category in categories
        ]
        test_exprs = [
            _categorical_expr(col, category, f"{col}_{category}", null_label)
            for category in categories
        ] if col in test.columns else [
            pl.lit(0, dtype=pl.UInt8).alias(f"{col}_{category}")
            for category in categories
        ]

        if train_exprs:
            train = train.with_columns(train_exprs)
            test = test.with_columns(test_exprs)
            dummy_cols += len(train_exprs)
        train = train.drop(col)
        test = test.drop(col, strict=False)

    logger.info("Encoded %s binary categorical columns and created %s dummy columns.", binary_cols, dummy_cols)
    return train, test


def apply_frequency_encoding(train: pl.DataFrame, test: pl.DataFrame, config):
    cat_config = config["pipeline"]["categorical_encoding"]
    fill_value = cat_config["frequency_unknown_value"]
    normalize = cat_config["frequency_normalize"]

    for col in FREQUENCY_ENCODING_COLS:
        if col not in train.columns:
            continue

        freq_col = f"{col}_FREQ"
        freqs = train.group_by(col).len().rename({"len": freq_col})
        if normalize:
            freqs = freqs.with_columns((pl.col(freq_col) / pl.lit(train.height)).alias(freq_col))

        train = (
            train.join(freqs, on=col, how="left")
            .with_columns(pl.col(freq_col).fill_null(fill_value))
            .drop(col)
        )
        test = (
            test.join(freqs, on=col, how="left")
            .with_columns(pl.col(freq_col).fill_null(fill_value))
            .drop(col)
        )

    return train, test
