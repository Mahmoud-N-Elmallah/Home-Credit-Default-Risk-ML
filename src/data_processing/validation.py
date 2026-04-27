from pathlib import Path

import polars as pl
import yaml


AUXILIARY_REQUIRED_COLUMNS = {
    "bureau": ["SK_ID_CURR", "SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU"],
    "previous_application": ["SK_ID_CURR", "SK_ID_PREV"],
    "pos_cash_balance": ["SK_ID_CURR", "SK_ID_PREV"],
    "installments_payments": ["SK_ID_CURR", "SK_ID_PREV"],
    "credit_card_balance": ["SK_ID_CURR", "SK_ID_PREV"],
}

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def validation_report_path(config):
    path = Path(config["data"]["final"]["validation_report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def check_result(name, passed, detail=None):
    return {"name": name, "passed": bool(passed), "detail": detail or {}}


def failed_checks(report):
    failures = []
    for section in report.values():
        for check in section:
            if not check["passed"]:
                failures.append(check["name"])
    return failures


def raise_if_failed(report, label):
    failures = failed_checks(report)
    if failures:
        raise ValueError(f"{label} validation failed: {', '.join(failures)}")


def validate_raw_paths(config):
    checks = []
    for name, path_value in config["data"]["raw"].items():
        path = Path(path_value)
        exists = path.exists()
        non_empty = exists and path.stat().st_size > 0
        checks.append(check_result(f"raw_path_exists.{name}", exists, {"path": str(path)}))
        checks.append(check_result(f"raw_path_non_empty.{name}", non_empty, {"path": str(path)}))
    report = {"raw_paths": checks}
    raise_if_failed(report, "Raw path")
    return report


def schema_names(frame):
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema().names()
    return frame.columns


def validate_required_columns(name, columns, required):
    missing = [col for col in required if col not in columns]
    return check_result(f"required_columns.{name}", not missing, {"missing": missing})


def validate_application_frames(train, test, config):
    id_col = config["training"]["id_col"]
    target_col = config["training"]["target_col"]
    checks = [
        validate_required_columns("application_train", train.columns, [id_col, target_col]),
        validate_required_columns("application_test", test.columns, [id_col]),
        check_result("target_absent.application_test", target_col not in test.columns),
        check_result("unique_id.application_train", train.get_column(id_col).n_unique() == train.height),
        check_result("unique_id.application_test", test.get_column(id_col).n_unique() == test.height),
    ]

    if target_col in train.columns:
        target_values = sorted(train.get_column(target_col).drop_nulls().unique().to_list())
        checks.append(check_result("binary_target.application_train", set(target_values).issubset({0, 1}), {"values": target_values}))
    return checks


def validate_auxiliary_frames(frames):
    checks = []
    for name, required in AUXILIARY_REQUIRED_COLUMNS.items():
        columns = schema_names(frames[name])
        checks.append(validate_required_columns(name, columns, required))
    return checks


def validate_raw_frames(train, test, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance, config):
    frames = {
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": prev_app,
        "pos_cash_balance": pos_cash,
        "installments_payments": installments,
        "credit_card_balance": cc_balance,
    }
    report = {
        "raw_schema": validate_application_frames(train, test, config) + validate_auxiliary_frames(frames),
    }
    raise_if_failed(report, "Raw schema")
    return report


def has_nulls(frame):
    return sum(frame.null_count().row(0)) > 0


def has_invalid_numeric_values(frame):
    numeric_cols = [col for col in frame.columns if frame[col].dtype in NUMERIC_DTYPES]
    if not numeric_cols:
        return False
    checks = [
        pl.col(col).cast(pl.Float64, strict=False).is_nan()
        | pl.col(col).cast(pl.Float64, strict=False).is_infinite()
        for col in numeric_cols
    ]
    return bool(frame.select(pl.any_horizontal(checks).any()).item())


def validate_final_frames(train, test, config):
    id_col = config["training"]["id_col"]
    target_col = config["training"]["target_col"]
    expected_test_cols = [col for col in train.columns if col != target_col]
    target_values = []
    if target_col in train.columns:
        target_values = sorted(train.get_column(target_col).drop_nulls().unique().to_list())

    report = {
        "final_schema": [
            check_result("target_present.final_train", target_col in train.columns),
            check_result("target_absent.final_test", target_col not in test.columns),
            check_result("binary_target.final_train", set(target_values).issubset({0, 1}), {"values": target_values}),
            check_result("unique_id.final_train", train.get_column(id_col).n_unique() == train.height),
            check_result("unique_id.final_test", test.get_column(id_col).n_unique() == test.height),
            check_result("feature_alignment.final_test", test.columns == expected_test_cols),
            check_result("no_nulls.final_train", not has_nulls(train)),
            check_result("no_nulls.final_test", not has_nulls(test)),
            check_result("no_invalid_numeric.final_train", not has_invalid_numeric_values(train)),
            check_result("no_invalid_numeric.final_test", not has_invalid_numeric_values(test)),
        ],
    }
    raise_if_failed(report, "Final data")
    return report


def write_validation_report(config, *reports):
    merged = {}
    for report in reports:
        merged.update(report)
    path = validation_report_path(config)
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(merged, file, sort_keys=False)
    return path
