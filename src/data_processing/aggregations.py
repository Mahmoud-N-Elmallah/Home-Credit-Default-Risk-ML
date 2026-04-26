import logging

import polars as pl

from src.data_processing.constants import (
    BUREAU,
    BUREAU_BALANCE,
    BUREAU_RECENCY_DAYS,
    CREDIT_CARD_BALANCE,
    INSTALLMENTS,
    LAST_N_RECORDS,
    POS_CASH,
    PREVIOUS_APPLICATION,
    RECENCY_DAYS,
    RECENCY_MONTHS,
)
from src.data_processing.encoding import get_proportions
from src.data_processing.features import (
    _available_lazy_columns,
    _existing_columns,
    _feature_enabled,
    _last_n_filter,
    _max_agg_exprs,
    _mean_agg_exprs,
    _safe_ratio_expr,
    _sum_agg_exprs,
    _trend_expr,
)


logger = logging.getLogger(__name__)


def _rename_with_prefix(df: pl.LazyFrame, id_col, prefix):
    rename_dict = {col: f"{prefix}{col}" for col in df.collect_schema().names() if col != id_col}
    return df.rename(rename_dict)


def _bureau_balance_agg(bureau_balance, bb_id, bb_status, bb_month):
    status_exprs = get_proportions(bureau_balance, bb_status, bb_status)
    bb_agg = bureau_balance.group_by(bb_id).agg(
        [
            pl.col(bb_month).count().alias(f"{bb_month}_count"),
            pl.col(bb_month).min().alias(f"{bb_month}_min"),
            pl.col(bb_month).max().alias(f"{bb_month}_max"),
        ] + status_exprs
    )
    for months in RECENCY_MONTHS:
        bb_recent = bureau_balance.filter(pl.col(bb_month) >= -months)
        status_exprs_recent = get_proportions(bb_recent, bb_status, f"{bb_status}_{months}M")
        bb_agg_recent = bb_recent.group_by(bb_id).agg(status_exprs_recent)
        bb_agg = bb_agg.join(bb_agg_recent, on=bb_id, how="left")
    return bb_agg


def _add_bureau_status_splits(b_agg, bureau, id_curr, bureau_cols, bureau_config):
    days_credit = bureau_config["days_credit_col"]
    amt_credit_sum = bureau_config["amt_credit_sum_col"]
    amt_credit_sum_debt = bureau_config["amt_credit_sum_debt_col"]
    debt_credit_ratio = bureau_config["debt_credit_ratio_col"]
    credit_active = bureau_config["credit_active_col"]
    split_exprs = [
        pl.col(days_credit).mean().alias(f"{days_credit}_mean"),
        pl.col(amt_credit_sum).sum().alias(f"{amt_credit_sum}_sum"),
        pl.col(amt_credit_sum_debt).sum().alias(f"{amt_credit_sum_debt}_sum"),
    ] + _mean_agg_exprs(_existing_columns([debt_credit_ratio], bureau_cols))
    for split_name, split_value in bureau_config["status_splits"].items():
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
    return b_agg


def _add_bureau_recency(b_agg, bureau, id_curr, bureau_cols, bureau_config):
    days_credit = bureau_config["days_credit_col"]
    amt_credit_sum = bureau_config["amt_credit_sum_col"]
    amt_credit_sum_debt = bureau_config["amt_credit_sum_debt_col"]
    debt_credit_ratio = bureau_config["debt_credit_ratio_col"]
    recency_cols = _existing_columns([amt_credit_sum, amt_credit_sum_debt, debt_credit_ratio], bureau_cols)
    for days in BUREAU_RECENCY_DAYS:
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
    return b_agg


def _add_previous_recency(agg, prev_app, id_curr, prev_config, interest_cols):
    prev_id = prev_config["id_col"]
    app_credit_gap = prev_config["app_credit_gap_col"]
    days_decision = prev_config["days_decision_col"]
    for days in RECENCY_DAYS:
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
    return agg


def _add_previous_status_values(agg, prev_app, id_curr, prev_config, interest_cols):
    prev_id = prev_config["id_col"]
    amt_credit = prev_config["amt_credit_col"]
    app_credit_gap = prev_config["app_credit_gap_col"]
    status_col = prev_config["status_col"]
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
    return agg


def _add_previous_last_n(agg, prev_app, id_curr, prev_config, interest_cols):
    prev_id = prev_config["id_col"]
    amt_credit = prev_config["amt_credit_col"]
    app_credit_gap = prev_config["app_credit_gap_col"]
    days_decision = prev_config["days_decision_col"]
    for n in LAST_N_RECORDS:
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
    return agg


def _clean_pos_cash(pos_cash, pos_cols, pos_config):
    clean_exprs = []
    count_exprs = []
    for col, min_value in pos_config["cleanup_min_values"].items():
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
        logger.info("POS cleanup adjusted configured invalid values: %s", counts)
    return pos_cash.with_columns(clean_exprs) if clean_exprs else pos_cash


def _add_pos_recency(agg, pos_cash, id_curr, pos_config, optional_cols):
    month_col = pos_config["month_col"]
    dpd_col = pos_config["dpd_col"]
    for months in RECENCY_MONTHS:
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
    return agg


def _add_pos_last_n(agg, pos_cash, id_curr, pos_config, optional_cols):
    month_col = pos_config["month_col"]
    dpd_col = pos_config["dpd_col"]
    for n in LAST_N_RECORDS:
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
    return agg


def _add_installment_recency(agg, installments, id_curr, inst_config):
    days_instalment = inst_config["days_instalment_col"]
    payment_perc = inst_config["payment_perc_col"]
    payment_diff = inst_config["payment_diff_col"]
    dpd_col = inst_config["dpd_col"]
    dbd_col = inst_config["dbd_col"]
    for days in RECENCY_DAYS:
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
    return agg


def _add_installment_last_n(agg, installments, id_curr, inst_config):
    days_instalment = inst_config["days_instalment_col"]
    payment_perc = inst_config["payment_perc_col"]
    payment_diff = inst_config["payment_diff_col"]
    dpd_col = inst_config["dpd_col"]
    dbd_col = inst_config["dbd_col"]
    for n in LAST_N_RECORDS:
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
    return agg


def _add_credit_card_recency(agg, cc_balance, id_curr, cc_config, optional_cols):
    month_col = cc_config["month_col"]
    drawings = cc_config["drawings_col"]
    dpd_col = cc_config["dpd_col"]
    utilization = cc_config["utilization_col"]
    for months in RECENCY_MONTHS:
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
    return agg


def _add_credit_card_last_n(agg, cc_balance, id_curr, cc_config, optional_cols):
    month_col = cc_config["month_col"]
    drawings = cc_config["drawings_col"]
    dpd_col = cc_config["dpd_col"]
    utilization = cc_config["utilization_col"]
    for n in LAST_N_RECORDS:
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
    return agg


def agg_bureau(bureau: pl.LazyFrame, bureau_balance: pl.LazyFrame, config):
    """Step 3.1 - Bureau aggregation."""
    logger.info("Aggregating bureau...")
    id_curr = config["training"]["id_col"]
    bb_config = BUREAU_BALANCE
    bureau_config = BUREAU
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

    bb_agg = _bureau_balance_agg(bureau_balance, bb_id, bb_status, bb_month)

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
        b_agg = _add_bureau_status_splits(b_agg, bureau, id_curr, bureau_cols, bureau_config)

    if _feature_enabled(config, "recency_windows"):
        b_agg = _add_bureau_recency(b_agg, bureau, id_curr, bureau_cols, bureau_config)

    return _rename_with_prefix(b_agg, id_curr, bureau_config["prefix"])


def agg_prev_app(prev_app: pl.LazyFrame, config):
    """Step 3.2 - Previous applications aggregation."""
    logger.info("Aggregating previous apps...")
    id_curr = config["training"]["id_col"]
    prev_config = PREVIOUS_APPLICATION
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
        agg = _add_previous_recency(agg, prev_app, id_curr, prev_config, interest_cols)

    if _feature_enabled(config, "previous_interest"):
        agg = _add_previous_status_values(agg, prev_app, id_curr, prev_config, interest_cols)

    if _feature_enabled(config, "last_n_aggregations"):
        agg = _add_previous_last_n(agg, prev_app, id_curr, prev_config, interest_cols)

    return _rename_with_prefix(agg, id_curr, prev_config["prefix"])


def agg_pos_cash(pos_cash: pl.LazyFrame, config):
    """Step 3.3 - POS cash balance aggregation."""
    logger.info("Aggregating POS cash...")
    id_curr = config["training"]["id_col"]
    pos_config = POS_CASH
    pos_cols = _available_lazy_columns(pos_cash)
    prev_id = pos_config["id_col"]
    month_col = pos_config["month_col"]
    dpd_col = pos_config["dpd_col"]
    cnt_instalment = pos_config["cnt_instalment_col"]
    cnt_instalment_future = pos_config["cnt_instalment_future_col"]

    if _feature_enabled(config, "pos_cash_cleanup"):
        pos_cash = _clean_pos_cash(pos_cash, pos_cols, pos_config)

    optional_cols = _existing_columns([cnt_instalment, cnt_instalment_future], pos_cols)

    agg = pos_cash.group_by(id_curr).agg(
        [
            pl.col(prev_id).n_unique().alias(f"{prev_id}_nunique"),
            pl.col(dpd_col).max().alias(f"{dpd_col}_max"),
            pl.col(dpd_col).mean().alias(f"{dpd_col}_mean"),
        ] + _mean_agg_exprs(optional_cols) + _max_agg_exprs(optional_cols)
    )

    if _feature_enabled(config, "recency_windows"):
        agg = _add_pos_recency(agg, pos_cash, id_curr, pos_config, optional_cols)

    if _feature_enabled(config, "last_n_aggregations"):
        agg = _add_pos_last_n(agg, pos_cash, id_curr, pos_config, optional_cols)

    return _rename_with_prefix(agg, id_curr, pos_config["prefix"])


def agg_installments(installments: pl.LazyFrame, config):
    """Step 3.4 - Installment payments aggregation."""
    logger.info("Aggregating installments...")
    id_curr = config["training"]["id_col"]
    eps = float(config["globals"]["division_epsilon"])
    inst_config = INSTALLMENTS
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
        agg = _add_installment_recency(agg, installments, id_curr, inst_config)

    if _feature_enabled(config, "last_n_aggregations"):
        agg = _add_installment_last_n(agg, installments, id_curr, inst_config)

    return _rename_with_prefix(agg, id_curr, inst_config["prefix"])


def agg_cc_balance(cc_balance: pl.LazyFrame, config):
    """Step 3.5 - Credit card balance aggregation."""
    logger.info("Aggregating CC balance...")
    id_curr = config["training"]["id_col"]
    eps = float(config["globals"]["division_epsilon"])
    cc_config = CREDIT_CARD_BALANCE
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
        agg = _add_credit_card_recency(agg, cc_balance, id_curr, cc_config, optional_cols)

    if _feature_enabled(config, "last_n_aggregations"):
        agg = _add_credit_card_last_n(agg, cc_balance, id_curr, cc_config, optional_cols)

    return _rename_with_prefix(agg, id_curr, cc_config["prefix"])
