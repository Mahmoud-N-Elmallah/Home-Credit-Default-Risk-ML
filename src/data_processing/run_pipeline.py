from pathlib import Path

import numpy as np
import polars as pl
import yaml


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
    output_features = fe_config["output_features"]
    app_cols = fe_config["application_source_cols"]
    days_birth_col = app_cols["days_birth"]
    goods_price_col = app_cols["amt_goods_price"]
    credit_col = app_cols["amt_credit"]
    days_employed_col = config["pipeline"]["anomaly_fix"]["days_employed_col"]
    anomaly_val = config["pipeline"]["anomaly_fix"]["days_employed"]
    eps = float(config["globals"]["division_epsilon"])
    day_unit = float(fe_config["day_unit"])

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
                    (pl.col(days_birth_col).cast(pl.Float64).abs() / pl.lit(day_unit))
                    .alias(output_features["age_years"])
                )
            down_payment_cols = [goods_price_col, credit_col]
            if all(col in df.columns for col in down_payment_cols):
                app_exprs.extend(
                    [
                        (
                            pl.col(goods_price_col).cast(pl.Float64)
                            - pl.col(credit_col).cast(pl.Float64)
                        ).alias(output_features["down_payment"]),
                        (
                            (
                                pl.col(goods_price_col).cast(pl.Float64)
                                - pl.col(credit_col).cast(pl.Float64)
                            )
                            / (pl.col(credit_col).cast(pl.Float64) + pl.lit(eps))
                        ).alias(output_features["down_payment_ratio"]),
                    ]
                )
            app_exprs.extend(
                [
                    _safe_ratio_expr(feature["numerator"], feature["denominator"], feature["name"], eps)
                    for feature in fe_config["extended_ratio_features"]
                    if all(col in df.columns for col in [feature["numerator"], feature["denominator"]])
                ]
            )
        if _feature_enabled(config, "missing_indicators"):
            app_exprs.extend(
                [
                    pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_is_missing")
                    for col in fe_config["missing_indicator_cols"]
                    if col in df.columns and f"{col}_is_missing" not in df.columns
                ]
            )
        if app_exprs:
            df = df.with_columns(app_exprs)

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
            df = df.with_columns(pl.sum_horizontal(doc_cols).alias(output_features["document_count"]))

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
    agg_config = config["pipeline"]["aggregations"]
    bb_config = agg_config["bureau_balance"]
    bureau_config = agg_config["bureau"]
    bureau_cols = _available_lazy_columns(bureau)
    bb_id = bb_config["id_col"]
    bb_status = bb_config["status_col"]
    bb_month = bb_config["month_col"]
    days_credit = bureau_config["days_credit_col"]
    amt_credit_sum = bureau_config["amt_credit_sum_col"]
    amt_credit_sum_debt = bureau_config["amt_credit_sum_debt_col"]
    amt_credit_sum_limit = bureau_config["amt_credit_sum_limit_col"]
    amt_credit_sum_overdue = bureau_config["amt_credit_sum_overdue_col"]
    days_credit_enddate = bureau_config["days_credit_enddate_col"]
    days_enddate_fact = bureau_config["days_enddate_fact_col"]
    debt_credit_ratio = bureau_config["debt_credit_ratio_col"]
    credit_active = bureau_config["credit_active_col"]
    status_splits = bureau_config["status_splits"]
    active_val = status_splits["active"]
    eps = float(config["globals"]["division_epsilon"])

    status_exprs = get_proportions(bureau_balance, bb_status, bb_status)
    bb_agg = bureau_balance.group_by(bb_id).agg(
        [
            pl.col(bb_month).count().alias(f"{bb_month}_count"),
            pl.col(bb_month).min().alias(f"{bb_month}_min"),
            pl.col(bb_month).max().alias(f"{bb_month}_max"),
        ] + status_exprs
    )

    for months in fe_config["recency_months"]:
        bb_recent = bureau_balance.filter(pl.col(bb_month) >= -months)
        status_exprs_recent = get_proportions(bb_recent, bb_status, f"{bb_status}_{months}M")
        bb_agg_recent = bb_recent.group_by(bb_id).agg(status_exprs_recent)
        bb_agg = bb_agg.join(bb_agg_recent, on=bb_id, how="left")

    if _feature_enabled(config, "bureau_extended") and all(
        col in bureau_cols for col in [amt_credit_sum_debt, amt_credit_sum]
    ):
        bureau = bureau.with_columns(_safe_ratio_expr(amt_credit_sum_debt, amt_credit_sum, debt_credit_ratio, eps))
        bureau_cols.add(debt_credit_ratio)

    bureau = bureau.join(bb_agg, on=bb_id, how="left")

    bb_agg_cols = [col for col in bb_agg.collect_schema().names() if col != bb_id]
    bb_mean_exprs = [pl.col(col).mean().alias(f"{col}_mean") for col in bb_agg_cols]
    optional_mean_cols = _existing_columns(
        [amt_credit_sum_limit, amt_credit_sum_overdue, days_credit_enddate, days_enddate_fact, debt_credit_ratio],
        bureau_cols,
    )

    b_agg = bureau.group_by(id_curr).agg(
        [
            pl.col(days_credit).min().alias(f"{days_credit}_min"),
            pl.col(days_credit).max().alias(f"{days_credit}_max"),
            pl.col(days_credit).mean().alias(f"{days_credit}_mean"),
            pl.col(amt_credit_sum).sum().alias(f"{amt_credit_sum}_sum"),
            pl.col(amt_credit_sum_debt).sum().alias(f"{amt_credit_sum_debt}_sum"),
            pl.col(credit_active).eq(active_val).sum().alias("ACTIVE_LOANS_COUNT"),
            pl.col(credit_active).eq(active_val).mean().alias(f"{credit_active}_prop_active"),
        ] + bb_mean_exprs + _mean_agg_exprs(optional_mean_cols) + _sum_agg_exprs(
            _existing_columns([amt_credit_sum_limit, amt_credit_sum_overdue], bureau_cols)
        )
    )

    if _feature_enabled(config, "bureau_extended"):
        split_exprs = [
            pl.col(days_credit).mean().alias(f"{days_credit}_mean"),
            pl.col(amt_credit_sum).sum().alias(f"{amt_credit_sum}_sum"),
            pl.col(amt_credit_sum_debt).sum().alias(f"{amt_credit_sum_debt}_sum"),
        ] + _mean_agg_exprs(_existing_columns([debt_credit_ratio], bureau_cols))
        for split_name, split_value in status_splits.items():
            split_prefix = split_name.upper()
            split_agg = (
                bureau.filter(pl.col(credit_active).eq(split_value))
                .group_by(id_curr)
                .agg([pl.len().alias(f"{split_prefix}_LOANS_COUNT")] + split_exprs)
            )
            split_rename = {
                col: f"{split_prefix}_{col}"
                for col in split_agg.collect_schema().names()
                if col != id_curr
            }
            b_agg = b_agg.join(split_agg.rename(split_rename), on=id_curr, how="left")

    if _feature_enabled(config, "recency_windows"):
        recency_cols = _existing_columns([amt_credit_sum, amt_credit_sum_debt, debt_credit_ratio], bureau_cols)
        for days in fe_config["bureau_recency_days"]:
            recent = bureau.filter(pl.col(days_credit) >= -days)
            recent_agg = recent.group_by(id_curr).agg(
                [pl.len().alias(f"BUREAU_COUNT_{days}D")]
                + _sum_agg_exprs(_existing_columns([amt_credit_sum, amt_credit_sum_debt], bureau_cols))
                + _mean_agg_exprs([col for col in recency_cols if col == debt_credit_ratio])
            )
            recent_rename = {
                col: f"{col}_{days}D"
                for col in recent_agg.collect_schema().names()
                if col != id_curr and not col.endswith(f"_{days}D")
            }
            b_agg = b_agg.join(recent_agg.rename(recent_rename), on=id_curr, how="left")

    rename_dict = {col: f"{bureau_config['prefix']}{col}" for col in b_agg.collect_schema().names() if col != id_curr}
    return b_agg.rename(rename_dict)


def agg_prev_app(prev_app: pl.LazyFrame, config):
    """Step 3.2 - Previous applications aggregation."""
    print("Aggregating previous apps...")
    id_curr = config["training"]["id_col"]
    fe_config = config["pipeline"]["feature_engineering"]
    prev_config = config["pipeline"]["aggregations"]["previous_application"]
    prev_cols = _available_lazy_columns(prev_app)
    prev_id = prev_config["id_col"]
    amt_application = prev_config["amt_application_col"]
    amt_credit = prev_config["amt_credit_col"]
    app_credit_gap = prev_config["app_credit_gap_col"]
    status_col = prev_config["status_col"]
    days_decision = prev_config["days_decision_col"]
    amt_annuity = prev_config["amt_annuity_col"]
    cnt_payment = prev_config["cnt_payment_col"]
    total_payment = prev_config["total_payment_col"]
    simple_interest = prev_config["simple_interest_col"]
    credit_to_annuity = prev_config["credit_to_annuity_col"]
    eps = float(config["globals"]["division_epsilon"])

    prev_app = prev_app.with_columns((pl.col(amt_application) - pl.col(amt_credit)).alias(app_credit_gap))
    prev_cols.add(app_credit_gap)
    if _feature_enabled(config, "previous_interest"):
        if all(col in prev_cols for col in [amt_annuity, cnt_payment]):
            prev_app = prev_app.with_columns((pl.col(amt_annuity) * pl.col(cnt_payment)).alias(total_payment))
            prev_cols.add(total_payment)
        interest_exprs = []
        if all(col in prev_cols for col in [total_payment, amt_credit]):
            interest_exprs.append((pl.col(total_payment) - pl.col(amt_credit)).alias(simple_interest))
            prev_cols.add(simple_interest)
        if all(col in prev_cols for col in [amt_credit, amt_annuity]):
            interest_exprs.append(_safe_ratio_expr(amt_credit, amt_annuity, credit_to_annuity, eps))
            prev_cols.add(credit_to_annuity)
        if interest_exprs:
            prev_app = prev_app.with_columns(interest_exprs)
    status_exprs = get_proportions(prev_app, status_col, status_col)
    interest_cols = _existing_columns([total_payment, simple_interest, credit_to_annuity], prev_cols)

    agg = prev_app.group_by(id_curr).agg(
        [
            pl.col(prev_id).count().alias(f"{prev_id}_count"),
            pl.col(amt_annuity).mean().alias(f"{amt_annuity}_mean"),
            pl.col(amt_credit).sum().alias(f"{amt_credit}_sum"),
            pl.col(app_credit_gap).sum().alias(f"{app_credit_gap}_sum"),
            pl.col(app_credit_gap).mean().alias(f"{app_credit_gap}_mean"),
            pl.col(days_decision).min().alias(f"{days_decision}_min"),
            pl.col(days_decision).mean().alias(f"{days_decision}_mean"),
        ] + status_exprs + _mean_agg_exprs(interest_cols)
    )

    if _feature_enabled(config, "recency_windows"):
        for days in fe_config["recency_days"]:
            recent = prev_app.filter(pl.col(days_decision) >= -days)
            recent_agg = recent.group_by(id_curr).agg(
                [
                    pl.col(prev_id).count().alias(f"{prev_id}_count_{days}D"),
                    pl.col(app_credit_gap).mean().alias(f"{app_credit_gap}_mean_{days}D"),
                ] + _mean_agg_exprs(interest_cols)
            )
            recent_rename = {
                col: f"{col}_{days}D"
                for col in recent_agg.collect_schema().names()
                if col != id_curr and not col.endswith(f"_{days}D")
            }
            agg = agg.join(recent_agg.rename(recent_rename), on=id_curr, how="left")

    if _feature_enabled(config, "previous_interest"):
        for label, value in prev_config["status_values"].items():
            status_agg = prev_app.filter(pl.col(status_col).eq(value)).group_by(id_curr).agg(
                [
                    pl.col(prev_id).count().alias(f"{label.upper()}_{prev_id}_count"),
                    pl.col(amt_credit).mean().alias(f"{label.upper()}_{amt_credit}_mean"),
                    pl.col(app_credit_gap).mean().alias(f"{label.upper()}_{app_credit_gap}_mean"),
                ] + _mean_agg_exprs(interest_cols)
            )
            status_rename = {
                col: f"{label.upper()}_{col}"
                for col in status_agg.collect_schema().names()
                if col != id_curr and not col.startswith(f"{label.upper()}_")
            }
            agg = agg.join(status_agg.rename(status_rename), on=id_curr, how="left")

    if _feature_enabled(config, "last_n_aggregations"):
        for n in fe_config["last_n_records"]:
            recent_n = _last_n_filter(prev_app, id_curr, days_decision, n)
            last_agg = recent_n.group_by(id_curr).agg(
                [
                    pl.col(prev_id).count().alias(f"LAST_{n}_{prev_id}_count"),
                    pl.col(amt_credit).mean().alias(f"LAST_{n}_{amt_credit}_mean"),
                    pl.col(app_credit_gap).mean().alias(f"LAST_{n}_{app_credit_gap}_mean"),
                ] + _mean_agg_exprs(interest_cols)
            )
            last_rename = {
                col: f"LAST_{n}_{col}"
                for col in last_agg.collect_schema().names()
                if col != id_curr and not col.startswith(f"LAST_{n}_")
            }
            agg = agg.join(last_agg.rename(last_rename), on=id_curr, how="left")

    rename_dict = {col: f"{prev_config['prefix']}{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)


def agg_pos_cash(pos_cash: pl.LazyFrame, config):
    """Step 3.3 - POS cash balance aggregation."""
    print("Aggregating POS cash...")
    id_curr = config["training"]["id_col"]
    fe_config = config["pipeline"]["feature_engineering"]
    pos_config = config["pipeline"]["aggregations"]["pos_cash"]
    pos_cols = _available_lazy_columns(pos_cash)
    prev_id = pos_config["id_col"]
    month_col = pos_config["month_col"]
    dpd_col = pos_config["dpd_col"]
    cnt_instalment = pos_config["cnt_instalment_col"]
    cnt_instalment_future = pos_config["cnt_instalment_future_col"]

    cleanup = pos_config.get("cleanup", {})
    if _feature_enabled(config, "pos_cash_cleanup") and cleanup.get("enabled", False):
        min_values = cleanup.get("min_values", {})
        clean_exprs = []
        count_exprs = []
        for col, min_value in min_values.items():
            if col in pos_cols:
                count_exprs.append((pl.col(col) < min_value).sum().alias(col))
                clean_exprs.append(
                    pl.when(pl.col(col) < min_value)
                    .then(pl.lit(min_value))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        if count_exprs:
            counts = pos_cash.select(count_exprs).collect().row(0, named=True)
            print(f"  POS cleanup adjusted configured invalid values: {counts}")
        if clean_exprs:
            pos_cash = pos_cash.with_columns(clean_exprs)

    optional_cols = _existing_columns([cnt_instalment, cnt_instalment_future], pos_cols)

    agg = pos_cash.group_by(id_curr).agg(
        [
            pl.col(prev_id).n_unique().alias(f"{prev_id}_nunique"),
            pl.col(dpd_col).max().alias(f"{dpd_col}_max"),
            pl.col(dpd_col).mean().alias(f"{dpd_col}_mean"),
        ] + _mean_agg_exprs(optional_cols) + _max_agg_exprs(optional_cols)
    )

    if _feature_enabled(config, "recency_windows"):
        for months in fe_config["recency_months"]:
            recent = pos_cash.filter(pl.col(month_col) >= -months)
            recent_agg = recent.group_by(id_curr).agg(
                [
                    pl.col(dpd_col).max().alias(f"{dpd_col}_max_{months}M"),
                    pl.col(dpd_col).mean().alias(f"{dpd_col}_mean_{months}M"),
                ] + _mean_agg_exprs(optional_cols)
            )
            recent_rename = {
                col: f"{col}_{months}M"
                for col in recent_agg.collect_schema().names()
                if col != id_curr and not col.endswith(f"_{months}M")
            }
            agg = agg.join(recent_agg.rename(recent_rename), on=id_curr, how="left")
            trend_col = f"{dpd_col}_mean_trend_{months}M"
            agg = agg.with_columns(_trend_expr(f"{dpd_col}_mean_{months}M", f"{dpd_col}_mean", trend_col))

    if _feature_enabled(config, "last_n_aggregations"):
        for n in fe_config["last_n_records"]:
            last_n = _last_n_filter(pos_cash, id_curr, month_col, n)
            last_agg = last_n.group_by(id_curr).agg(
                [
                    pl.col(dpd_col).max().alias(f"LAST_{n}_{dpd_col}_max"),
                    pl.col(dpd_col).mean().alias(f"LAST_{n}_{dpd_col}_mean"),
                ] + _mean_agg_exprs(optional_cols)
            )
            last_rename = {
                col: f"LAST_{n}_{col}"
                for col in last_agg.collect_schema().names()
                if col != id_curr and not col.startswith(f"LAST_{n}_")
            }
            agg = agg.join(last_agg.rename(last_rename), on=id_curr, how="left")

    rename_dict = {col: f"{pos_config['prefix']}{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)


def agg_installments(installments: pl.LazyFrame, config):
    """Step 3.4 - Installment payments aggregation."""
    print("Aggregating installments...")
    id_curr = config["training"]["id_col"]
    eps = float(config["globals"]["division_epsilon"])
    fe_config = config["pipeline"]["feature_engineering"]
    inst_config = config["pipeline"]["aggregations"]["installments"]
    instalment_version = inst_config["instalment_version_col"]
    amt_payment = inst_config["amt_payment_col"]
    amt_instalment = inst_config["amt_instalment_col"]
    days_entry_payment = inst_config["days_entry_payment_col"]
    days_instalment = inst_config["days_instalment_col"]
    payment_perc = inst_config["payment_perc_col"]
    payment_diff = inst_config["payment_diff_col"]
    dpd_col = inst_config["dpd_col"]
    dbd_col = inst_config["dbd_col"]

    installments = installments.with_columns(
        [
            (
                pl.col(amt_payment).cast(pl.Float64)
                / (pl.col(amt_instalment).cast(pl.Float64) + pl.lit(eps))
            ).alias(payment_perc),
            (
                pl.col(amt_instalment).cast(pl.Float64)
                - pl.col(amt_payment).cast(pl.Float64)
            ).alias(payment_diff),
            (
                pl.col(days_entry_payment).cast(pl.Float64)
                - pl.col(days_instalment).cast(pl.Float64)
            ).clip(lower_bound=0).alias(dpd_col),
            (
                pl.col(days_instalment).cast(pl.Float64)
                - pl.col(days_entry_payment).cast(pl.Float64)
            ).clip(lower_bound=0).alias(dbd_col),
        ]
    )

    agg = installments.group_by(id_curr).agg(
        [
            pl.col(instalment_version).n_unique().alias(f"{instalment_version}_nunique"),
            pl.col(dpd_col).max().alias(f"{dpd_col}_max"),
            pl.col(dpd_col).mean().alias(f"{dpd_col}_mean"),
            pl.col(dbd_col).max().alias(f"{dbd_col}_max"),
            pl.col(dbd_col).mean().alias(f"{dbd_col}_mean"),
            pl.col(payment_perc).mean().alias(f"{payment_perc}_mean"),
            pl.col(payment_diff).sum().alias(f"{payment_diff}_sum"),
            pl.col(payment_diff).mean().alias(f"{payment_diff}_mean"),
            pl.col(amt_instalment).sum().alias(f"{amt_instalment}_sum"),
            pl.col(amt_payment).sum().alias(f"{amt_payment}_sum"),
        ]
    )

    if _feature_enabled(config, "recency_windows"):
        for days in fe_config["recency_days"]:
            recent = installments.filter(pl.col(days_instalment) >= -days)
            recent_agg = recent.group_by(id_curr).agg(
                [
                    pl.col(dpd_col).mean().alias(f"{dpd_col}_mean_{days}D"),
                    pl.col(dbd_col).mean().alias(f"{dbd_col}_mean_{days}D"),
                    pl.col(payment_perc).mean().alias(f"{payment_perc}_mean_{days}D"),
                    pl.col(payment_diff).mean().alias(f"{payment_diff}_mean_{days}D"),
                ]
            )
            agg = agg.join(recent_agg, on=id_curr, how="left").with_columns(
                [
                    _trend_expr(f"{dpd_col}_mean_{days}D", f"{dpd_col}_mean", f"{dpd_col}_mean_trend_{days}D"),
                    _trend_expr(
                        f"{payment_perc}_mean_{days}D",
                        f"{payment_perc}_mean",
                        f"{payment_perc}_mean_trend_{days}D",
                    ),
                ]
            )

    if _feature_enabled(config, "last_n_aggregations"):
        for n in fe_config["last_n_records"]:
            last_n = _last_n_filter(installments, id_curr, days_instalment, n)
            last_agg = last_n.group_by(id_curr).agg(
                [
                    pl.col(dpd_col).mean().alias(f"LAST_{n}_{dpd_col}_mean"),
                    pl.col(dbd_col).mean().alias(f"LAST_{n}_{dbd_col}_mean"),
                    pl.col(payment_perc).mean().alias(f"LAST_{n}_{payment_perc}_mean"),
                    pl.col(payment_diff).mean().alias(f"LAST_{n}_{payment_diff}_mean"),
                ]
            )
            agg = agg.join(last_agg, on=id_curr, how="left")

    rename_dict = {col: f"{inst_config['prefix']}{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)


def agg_cc_balance(cc_balance: pl.LazyFrame, config):
    """Step 3.5 - Credit card balance aggregation."""
    print("Aggregating CC balance...")
    id_curr = config["training"]["id_col"]
    eps = float(config["globals"]["division_epsilon"])
    fe_config = config["pipeline"]["feature_engineering"]
    cc_config = config["pipeline"]["aggregations"]["credit_card_balance"]
    cc_cols = _available_lazy_columns(cc_balance)
    month_col = cc_config["month_col"]
    amt_balance = cc_config["amt_balance_col"]
    credit_limit = cc_config["credit_limit_col"]
    drawings = cc_config["drawings_col"]
    dpd_col = cc_config["dpd_col"]
    utilization = cc_config["utilization_col"]
    total_receivable = cc_config["total_receivable_col"]
    receivable_principal = cc_config["receivable_principal_col"]
    drawings_count = cc_config["drawings_count_col"]
    optional_cols = _existing_columns([total_receivable, receivable_principal, drawings_count], cc_cols)

    cc_balance = cc_balance.with_columns(
        (
            pl.col(amt_balance).cast(pl.Float64)
            / (pl.col(credit_limit).cast(pl.Float64) + pl.lit(eps))
        ).clip(0, 1).alias(utilization)
    )

    agg = cc_balance.group_by(id_curr).agg(
        [
            pl.col(amt_balance).mean().alias(f"{amt_balance}_mean"),
            pl.col(drawings).sum().alias(f"{drawings}_sum"),
            pl.col(utilization).mean().alias(f"{utilization}_mean"),
            pl.col(utilization).max().alias(f"{utilization}_max"),
            pl.col(dpd_col).max().alias(f"{dpd_col}_max"),
        ] + _mean_agg_exprs(optional_cols) + _sum_agg_exprs(optional_cols)
    )

    if _feature_enabled(config, "recency_windows"):
        for months in fe_config["recency_months"]:
            recent = cc_balance.filter(pl.col(month_col) >= -months)
            recent_agg = recent.group_by(id_curr).agg(
                [
                    pl.col(drawings).sum().alias(f"{drawings}_sum_{months}M"),
                    pl.col(utilization).mean().alias(f"{utilization}_mean_{months}M"),
                    pl.col(dpd_col).max().alias(f"{dpd_col}_max_{months}M"),
                ] + _mean_agg_exprs(optional_cols)
            )
            recent_rename = {
                col: f"{col}_{months}M"
                for col in recent_agg.collect_schema().names()
                if col != id_curr and not col.endswith(f"_{months}M")
            }
            agg = agg.join(recent_agg.rename(recent_rename), on=id_curr, how="left").with_columns(
                _trend_expr(f"{utilization}_mean_{months}M", f"{utilization}_mean", f"{utilization}_mean_trend_{months}M")
            )

    if _feature_enabled(config, "last_n_aggregations"):
        for n in fe_config["last_n_records"]:
            last_n = _last_n_filter(cc_balance, id_curr, month_col, n)
            last_agg = last_n.group_by(id_curr).agg(
                [
                    pl.col(drawings).sum().alias(f"LAST_{n}_{drawings}_sum"),
                    pl.col(utilization).mean().alias(f"LAST_{n}_{utilization}_mean"),
                    pl.col(dpd_col).max().alias(f"LAST_{n}_{dpd_col}_max"),
                ] + _mean_agg_exprs(optional_cols)
            )
            last_rename = {
                col: f"LAST_{n}_{col}"
                for col in last_agg.collect_schema().names()
                if col != id_curr and not col.startswith(f"LAST_{n}_")
            }
            agg = agg.join(last_agg.rename(last_rename), on=id_curr, how="left")

    rename_dict = {col: f"{cc_config['prefix']}{col}" for col in agg.collect_schema().names() if col != id_curr}
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
    aux_prefixes = [
        agg["prefix"]
        for agg in config["pipeline"]["aggregations"].values()
        if "prefix" in agg
    ]
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
    output_features = fe_config["output_features"]
    fill_value = config["pipeline"]["fill_values"]["generated_missing"]

    exprs = [
        _safe_ratio_expr(feature["numerator"], feature["denominator"], feature["name"], eps)
        for feature in fe_config["global_ratio_features"]
        if all(col in df.columns for col in [feature["numerator"], feature["denominator"]])
    ]

    ext_cols = fe_config["ext_sources"]
    if all(col in df.columns for col in ext_cols):
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in ext_cols])
        exprs.append(pl.mean_horizontal(ext_cols).alias(output_features["ext_sources_mean"]))
        exprs.append(pl.min_horizontal(ext_cols).alias(output_features["ext_sources_min"]))
        exprs.append(pl.max_horizontal(ext_cols).alias(output_features["ext_sources_max"]))
        exprs.append(
            (pl.max_horizontal(ext_cols) - pl.min_horizontal(ext_cols))
            .alias(output_features["ext_sources_range"])
        )
        exprs.append(
            (pl.col(ext_cols[0]) * pl.col(ext_cols[1]) * pl.col(ext_cols[2]))
            .alias(output_features["ext_sources_prod"])
        )
        mean = pl.mean_horizontal(ext_cols)
        exprs.append(
            pl.mean_horizontal([(pl.col(col) - mean) ** 2 for col in ext_cols])
            .sqrt()
            .alias(output_features["ext_sources_std"])
        )
        for base_col in fe_config["ext_source_interaction_bases"]:
            if base_col in df.columns:
                for ext_col in ext_cols:
                    exprs.append(_safe_ratio_expr(base_col, ext_col, f"{base_col}_TO_{ext_col}", eps))

    enq_cols = fe_config["enquiry_cols"]
    if all(col in df.columns for col in enq_cols):
        df = df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in enq_cols])
        exprs.append(pl.sum_horizontal(enq_cols).alias(output_features["enquiry_total"]))

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
    cleanup_info = {
        "dropped_correlated": sorted(to_drop_corr),
        "dropped_low_variance": sorted(to_drop_var),
        "missing_columns_added_to_test": sorted(missing_in_test),
    }
    return train, test, cleanup_info


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


def write_feature_manifest(train: pl.DataFrame, test: pl.DataFrame, config, cleanup_info):
    final_paths = config["data"]["final"]
    manifest_path = Path(final_paths["feature_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    target_col = config["training"]["target_col"]
    id_col = config["training"]["id_col"]
    feature_cols = [col for col in train.columns if col not in [target_col, id_col]]
    manifest = {
        "enabled_feature_sets": config["pipeline"]["feature_engineering"]["enabled_sets"],
        "source_tables_used": sorted(config["data"]["raw"].keys()),
        "final_train_shape": list(train.shape),
        "final_test_shape": list(test.shape),
        "final_feature_count": len(feature_cols),
        "train_columns": train.columns,
        "test_columns": test.columns,
        "dropped_columns": cleanup_info,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    print(f"Feature manifest saved to {manifest_path}")


def latest_submission_path(config):
    artifact_paths = config["training"]["artifact_paths"]
    models_dir = Path(artifact_paths["models_dir"])
    latest_path = Path(artifact_paths.get("latest_experiment", "latest_experiment.txt"))
    if not latest_path.is_absolute():
        latest_path = models_dir / latest_path
    if not latest_path.exists():
        return None

    experiment_dir = Path(latest_path.read_text(encoding="utf-8").strip())
    submission_path = Path(artifact_paths["submission"])
    return submission_path if submission_path.is_absolute() else experiment_dir / submission_path


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
    train_full, test_full, cleanup_info = feature_cleanup(train_full, test_full, config)
    validate(train_full, test_full, config)
    write_feature_manifest(train_full, test_full, config, cleanup_info)

    final_paths = config["data"]["final"]
    Path(final_paths["train"]).parent.mkdir(parents=True, exist_ok=True)
    train_full.write_csv(final_paths["train"])
    test_full.write_csv(final_paths["test"])
    submission_path = latest_submission_path(config)
    if config["pipeline"]["warn_on_stale_submission"] and submission_path is not None and submission_path.exists():
        print(f"Warning: existing submission may be stale after processing. Regenerate it with --train: {submission_path}")
    print("Data processing done.")
