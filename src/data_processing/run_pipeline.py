from pathlib import Path

import numpy as np
import polars as pl


POLARS_DTYPES = {
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "UInt8": pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "String": pl.String,
    "Boolean": pl.Boolean,
}


def _polars_dtype(dtype_name: str):
    try:
        return POLARS_DTYPES[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Polars dtype in config: {dtype_name}") from exc


def _csv_options(config):
    csv_config = config["data"]["csv"]
    schema_overrides = {
        col: _polars_dtype(dtype_name)
        for col, dtype_name in csv_config.get("schema_overrides", {}).items()
    }
    return {
        "infer_schema_length": csv_config["infer_schema_length"],
        "schema_overrides": schema_overrides,
        "null_values": csv_config["null_values"],
        "ignore_errors": csv_config["ignore_errors"],
    }


def _safe_ratio_expr(numerator, denominator, output_name, eps):
    return (
        pl.col(numerator).cast(pl.Float64)
        / (pl.col(denominator).cast(pl.Float64) + pl.lit(eps))
    ).alias(output_name)


def load_data(raw_paths, config):
    """Step 0 - Load & organize source files."""
    print("Loading data...")
    csv_options = _csv_options(config)

    train_base = pl.read_csv(raw_paths["application_train"], **csv_options)
    test_base = pl.read_csv(raw_paths["application_test"], **csv_options)

    bureau = pl.scan_csv(raw_paths["bureau"], **csv_options)
    bureau_balance = pl.scan_csv(raw_paths["bureau_balance"], **csv_options)
    prev_app = pl.scan_csv(raw_paths["previous_application"], **csv_options)
    pos_cash = pl.scan_csv(raw_paths["pos_cash_balance"], **csv_options)
    installments = pl.scan_csv(raw_paths["installments_payments"], **csv_options)
    cc_balance = pl.scan_csv(raw_paths["credit_card_balance"], **csv_options)

    return train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance


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

    print(f"  Encoded {binary_cols} binary categorical columns and created {dummy_cols} dummy columns.")
    return train, test


def apply_frequency_encoding(train: pl.DataFrame, test: pl.DataFrame, config):
    fe_config = config["pipeline"]["feature_engineering"]
    cat_config = config["pipeline"]["categorical_encoding"]
    fill_value = cat_config["frequency_unknown_value"]
    normalize = cat_config["frequency_normalize"]

    for col in fe_config["frequency_encoding_cols"]:
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


def preprocess_base(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 2 - Preprocess base application tables."""
    print("Preprocessing base tables...")
    fe_config = config["pipeline"]["feature_engineering"]
    anomaly_val = config["pipeline"]["anomaly_fix"]["days_employed"]
    eps = float(config["globals"]["division_epsilon"])

    def transform_base(df: pl.DataFrame):
        if "DAYS_EMPLOYED" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("DAYS_EMPLOYED") == anomaly_val)
                .then(None)
                .otherwise(pl.col("DAYS_EMPLOYED"))
                .alias("DAYS_EMPLOYED")
            )

        ratio_exprs = [
            _safe_ratio_expr(feature["numerator"], feature["denominator"], feature["name"], eps)
            for feature in fe_config["base_ratio_features"]
            if all(col in df.columns for col in [feature["numerator"], feature["denominator"]])
        ]
        if ratio_exprs:
            df = df.with_columns(ratio_exprs)

        doc_prefix = fe_config["document_flag_prefix"]
        doc_cols = [col for col in df.columns if col.startswith(doc_prefix)]
        if doc_cols:
            df = df.with_columns(pl.sum_horizontal(doc_cols).alias("DOCUMENT_COUNT"))

        return df

    train = transform_base(train)
    test = transform_base(test)
    train, test = apply_frequency_encoding(train, test, config)
    train, test = encode_categoricals(train, test, config)
    return train, test


def agg_bureau(bureau: pl.LazyFrame, bureau_balance: pl.LazyFrame, config):
    """Step 3.1 - Bureau aggregation."""
    print("Aggregating bureau...")
    id_curr = config["training"]["id_col"]
    fe_config = config["pipeline"]["feature_engineering"]

    status_exprs = get_proportions(bureau_balance, "STATUS", "STATUS")
    bb_agg = bureau_balance.group_by("SK_ID_BUREAU").agg(
        [
            pl.col("MONTHS_BALANCE").count().alias("MONTHS_BALANCE_count"),
            pl.col("MONTHS_BALANCE").min().alias("MONTHS_BALANCE_min"),
            pl.col("MONTHS_BALANCE").max().alias("MONTHS_BALANCE_max"),
        ] + status_exprs
    )

    for months in fe_config["recency_months"]:
        bb_recent = bureau_balance.filter(pl.col("MONTHS_BALANCE") >= -months)
        status_exprs_recent = get_proportions(bb_recent, "STATUS", f"STATUS_{months}M")
        bb_agg_recent = bb_recent.group_by("SK_ID_BUREAU").agg(status_exprs_recent)
        bb_agg = bb_agg.join(bb_agg_recent, on="SK_ID_BUREAU", how="left")

    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU", how="left")

    bb_agg_cols = [col for col in bb_agg.collect_schema().names() if col != "SK_ID_BUREAU"]
    bb_mean_exprs = [pl.col(col).mean().alias(f"{col}_mean") for col in bb_agg_cols]
    active_val = fe_config["categories"]["bureau_active"]

    b_agg = bureau.group_by(id_curr).agg(
        [
            pl.col("DAYS_CREDIT").min().alias("DAYS_CREDIT_min"),
            pl.col("DAYS_CREDIT").max().alias("DAYS_CREDIT_max"),
            pl.col("DAYS_CREDIT").mean().alias("DAYS_CREDIT_mean"),
            pl.col("AMT_CREDIT_SUM").sum().alias("AMT_CREDIT_SUM_sum"),
            pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("AMT_CREDIT_SUM_DEBT_sum"),
            pl.col("CREDIT_ACTIVE").eq(active_val).sum().alias("ACTIVE_LOANS_COUNT"),
            pl.col("CREDIT_ACTIVE").eq(active_val).mean().alias("CREDIT_ACTIVE_prop_active"),
        ] + bb_mean_exprs
    )

    rename_dict = {col: f"bureau_{col}" for col in b_agg.collect_schema().names() if col != id_curr}
    return b_agg.rename(rename_dict)


def agg_prev_app(prev_app: pl.LazyFrame, config):
    """Step 3.2 - Previous applications aggregation."""
    print("Aggregating previous apps...")
    id_curr = config["training"]["id_col"]
    fe_config = config["pipeline"]["feature_engineering"]

    prev_app = prev_app.with_columns(
        (pl.col("AMT_APPLICATION") - pl.col("AMT_CREDIT")).alias("APP_CREDIT_GAP")
    )
    status_exprs = get_proportions(prev_app, "NAME_CONTRACT_STATUS", "NAME_CONTRACT_STATUS")

    agg = prev_app.group_by(id_curr).agg(
        [
            pl.col("SK_ID_PREV").count().alias("SK_ID_PREV_count"),
            pl.col("AMT_ANNUITY").mean().alias("AMT_ANNUITY_mean"),
            pl.col("AMT_CREDIT").sum().alias("AMT_CREDIT_sum"),
            pl.col("APP_CREDIT_GAP").sum().alias("APP_CREDIT_GAP_sum"),
            pl.col("APP_CREDIT_GAP").mean().alias("APP_CREDIT_GAP_mean"),
            pl.col("DAYS_DECISION").min().alias("DAYS_DECISION_min"),
            pl.col("DAYS_DECISION").mean().alias("DAYS_DECISION_mean"),
        ] + status_exprs
    )

    for days in fe_config["recency_days_prev"]:
        recent = prev_app.filter(pl.col("DAYS_DECISION") >= -days)
        recent_agg = recent.group_by(id_curr).agg(
            [
                pl.col("SK_ID_PREV").count().alias(f"SK_ID_PREV_count_{days}D"),
                pl.col("APP_CREDIT_GAP").mean().alias(f"APP_CREDIT_GAP_mean_{days}D"),
            ]
        )
        agg = agg.join(recent_agg, on=id_curr, how="left")

    rename_dict = {col: f"prev_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)


def agg_pos_cash(pos_cash: pl.LazyFrame, config):
    """Step 3.3 - POS cash balance aggregation."""
    print("Aggregating POS cash...")
    id_curr = config["training"]["id_col"]
    fe_config = config["pipeline"]["feature_engineering"]

    agg = pos_cash.group_by(id_curr).agg(
        [
            pl.col("SK_ID_PREV").n_unique().alias("SK_ID_PREV_nunique"),
            pl.col("SK_DPD").max().alias("SK_DPD_max"),
            pl.col("SK_DPD").mean().alias("SK_DPD_mean"),
        ]
    )

    for months in fe_config["recency_months"]:
        recent = pos_cash.filter(pl.col("MONTHS_BALANCE") >= -months)
        recent_agg = recent.group_by(id_curr).agg(
            [
                pl.col("SK_DPD").max().alias(f"SK_DPD_max_{months}M"),
                pl.col("SK_DPD").mean().alias(f"SK_DPD_mean_{months}M"),
            ]
        )
        agg = agg.join(recent_agg, on=id_curr, how="left")

    rename_dict = {col: f"pos_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)


def agg_installments(installments: pl.LazyFrame, config):
    """Step 3.4 - Installment payments aggregation."""
    print("Aggregating installments...")
    id_curr = config["training"]["id_col"]
    eps = float(config["globals"]["division_epsilon"])

    installments = installments.with_columns(
        [
            (
                pl.col("AMT_PAYMENT").cast(pl.Float64)
                / (pl.col("AMT_INSTALMENT").cast(pl.Float64) + pl.lit(eps))
            ).alias("PAYMENT_PERC"),
            (
                pl.col("AMT_INSTALMENT").cast(pl.Float64)
                - pl.col("AMT_PAYMENT").cast(pl.Float64)
            ).alias("PAYMENT_DIFF"),
            (
                pl.col("DAYS_ENTRY_PAYMENT").cast(pl.Float64)
                - pl.col("DAYS_INSTALMENT").cast(pl.Float64)
            ).clip(lower_bound=0).alias("DPD"),
            (
                pl.col("DAYS_INSTALMENT").cast(pl.Float64)
                - pl.col("DAYS_ENTRY_PAYMENT").cast(pl.Float64)
            ).clip(lower_bound=0).alias("DBD"),
        ]
    )

    agg = installments.group_by(id_curr).agg(
        [
            pl.col("NUM_INSTALMENT_VERSION").n_unique().alias("NUM_INSTALMENT_VERSION_nunique"),
            pl.col("DPD").max().alias("DPD_max"),
            pl.col("DPD").mean().alias("DPD_mean"),
            pl.col("DBD").max().alias("DBD_max"),
            pl.col("DBD").mean().alias("DBD_mean"),
            pl.col("PAYMENT_PERC").mean().alias("PAYMENT_PERC_mean"),
            pl.col("PAYMENT_DIFF").sum().alias("PAYMENT_DIFF_sum"),
            pl.col("PAYMENT_DIFF").mean().alias("PAYMENT_DIFF_mean"),
            pl.col("AMT_INSTALMENT").sum().alias("AMT_INSTALMENT_sum"),
            pl.col("AMT_PAYMENT").sum().alias("AMT_PAYMENT_sum"),
        ]
    )
    rename_dict = {col: f"inst_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)


def agg_cc_balance(cc_balance: pl.LazyFrame, config):
    """Step 3.5 - Credit card balance aggregation."""
    print("Aggregating CC balance...")
    id_curr = config["training"]["id_col"]
    eps = float(config["globals"]["division_epsilon"])
    fe_config = config["pipeline"]["feature_engineering"]

    cc_balance = cc_balance.with_columns(
        (
            pl.col("AMT_BALANCE").cast(pl.Float64)
            / (pl.col("AMT_CREDIT_LIMIT_ACTUAL").cast(pl.Float64) + pl.lit(eps))
        ).clip(0, 1).alias("UTILIZATION")
    )

    agg = cc_balance.group_by(id_curr).agg(
        [
            pl.col("AMT_BALANCE").mean().alias("AMT_BALANCE_mean"),
            pl.col("AMT_DRAWINGS_CURRENT").sum().alias("AMT_DRAWINGS_CURRENT_sum"),
            pl.col("UTILIZATION").mean().alias("UTILIZATION_mean"),
            pl.col("UTILIZATION").max().alias("UTILIZATION_max"),
            pl.col("SK_DPD").max().alias("SK_DPD_max"),
        ]
    )

    for months in fe_config["recency_months"]:
        recent = cc_balance.filter(pl.col("MONTHS_BALANCE") >= -months)
        recent_agg = recent.group_by(id_curr).agg(
            [
                pl.col("AMT_DRAWINGS_CURRENT").sum().alias(f"AMT_DRAWINGS_CURRENT_sum_{months}M"),
                pl.col("UTILIZATION").mean().alias(f"UTILIZATION_mean_{months}M"),
                pl.col("SK_DPD").max().alias(f"SK_DPD_max_{months}M"),
            ]
        )
        agg = agg.join(recent_agg, on=id_curr, how="left")

    rename_dict = {col: f"cc_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)


def merge_all(base: pl.DataFrame, aggs: dict, config, name=""):
    """Step 4 - Sequential merge."""
    print(f"Merging tables for {name}...")
    id_curr = config["training"]["id_col"]
    df = base.lazy()
    for _, agg_df in aggs.items():
        df = df.join(agg_df, on=id_curr, how="left")

    df = df.collect()
    print(f"  Shape after all merges: {df.shape}")
    high_null_threshold = config["pipeline"]["high_null_threshold"]
    null_rates = [df.get_column(col).null_count() / df.height for col in df.columns]
    high_nulls = [
        col for col, rate in zip(df.columns, null_rates)
        if rate > high_null_threshold and col not in base.columns
    ]
    if high_nulls:
        print(f"  Warning: {len(high_nulls)} columns with high null rate (>{high_null_threshold}).")
    return df


def impute_missing(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 5 - Missing value imputation."""
    print("Imputing missing values...")
    fill_values = config["pipeline"]["fill_values"]
    aux_prefixes = config["pipeline"]["aux_table_prefixes"]
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
    print("Adding global features...")
    eps = float(config["globals"]["division_epsilon"])
    fe_config = config["pipeline"]["feature_engineering"]
    fill_value = config["pipeline"]["fill_values"]["generated_missing"]

    exprs = [
        _safe_ratio_expr(feature["numerator"], feature["denominator"], feature["name"], eps)
        for feature in fe_config["global_ratio_features"]
        if all(col in df.columns for col in [feature["numerator"], feature["denominator"]])
    ]

    ext_cols = fe_config["ext_sources"]
    if all(col in df.columns for col in ext_cols):
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in ext_cols])
        exprs.append(pl.mean_horizontal(ext_cols).alias("EXT_SOURCES_MEAN"))
        exprs.append((pl.col(ext_cols[0]) * pl.col(ext_cols[1]) * pl.col(ext_cols[2])).alias("EXT_SOURCES_PROD"))
        mean = pl.mean_horizontal(ext_cols)
        exprs.append(
            pl.mean_horizontal([(pl.col(col) - mean) ** 2 for col in ext_cols])
            .sqrt()
            .alias("EXT_SOURCES_STD")
        )

    enq_cols = fe_config["enquiry_cols"]
    if all(col in df.columns for col in enq_cols):
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in enq_cols])
        exprs.append(pl.sum_horizontal(enq_cols).alias("AMT_REQ_CREDIT_BUREAU_TOTAL"))

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
    print("Cleaning up features...")
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
    print(f"  Dropping {len(to_drop_corr)} correlated cols")
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

    print(f"  Dropping {len(to_drop_var)} low var cols")
    train = train.drop(to_drop_var, strict=False)

    test_cols = [col for col in train.columns if col != target_col]
    missing_in_test = [col for col in test_cols if col not in test.columns]
    if missing_in_test:
        test = test.with_columns([pl.lit(fill_value).alias(col) for col in missing_in_test])
    test = test.select(test_cols)
    return train, test


def validate(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 8 - Validation."""
    print("Validating...")
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

    print(f"Validation passed. Train: {train.shape}, Test: {test.shape}")


def run_pipeline(config):
    train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance = load_data(
        config["data"]["raw"],
        config,
    )
    train_base, test_base = preprocess_base(train_base, test_base, config)
    b_agg = agg_bureau(bureau, bureau_balance, config)
    p_agg = agg_prev_app(prev_app, config)
    pos_agg = agg_pos_cash(pos_cash, config)
    inst_agg = agg_installments(installments, config)
    cc_agg = agg_cc_balance(cc_balance, config)
    aggs = {"bureau": b_agg, "prev": p_agg, "pos": pos_agg, "inst": inst_agg, "cc": cc_agg}

    train_full = merge_all(train_base, aggs, config, "Train")
    test_full = merge_all(test_base, aggs, config, "Test")
    train_full, test_full = impute_missing(train_full, test_full, config)
    train_full = add_global_features(train_full, config)
    test_full = add_global_features(test_full, config)
    train_full, test_full = feature_cleanup(train_full, test_full, config)
    validate(train_full, test_full, config)

    final_paths = config["data"]["final"]
    Path(final_paths["train"]).parent.mkdir(parents=True, exist_ok=True)
    train_full.write_csv(final_paths["train"])
    test_full.write_csv(final_paths["test"])
    submission_path = Path(config["training"]["artifact_paths"]["submission"])
    if config["pipeline"]["warn_on_stale_submission"] and submission_path.exists():
        print(f"Warning: existing submission may be stale after processing. Regenerate it with --train: {submission_path}")
    print("Data processing done.")
