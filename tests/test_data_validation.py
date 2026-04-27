import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.data_processing.validation import (
    validate_final_frames,
    validate_raw_frames,
    validate_raw_paths,
    write_validation_report,
)


def config(tmp_dir=None):
    raw_dir = Path(tmp_dir or ".")
    return {
        "data": {
            "raw": {
                "application_train": str(raw_dir / "application_train.csv"),
                "application_test": str(raw_dir / "application_test.csv"),
                "bureau": str(raw_dir / "bureau.csv"),
                "bureau_balance": str(raw_dir / "bureau_balance.csv"),
                "previous_application": str(raw_dir / "previous_application.csv"),
                "pos_cash_balance": str(raw_dir / "POS_CASH_balance.csv"),
                "installments_payments": str(raw_dir / "installments_payments.csv"),
                "credit_card_balance": str(raw_dir / "credit_card_balance.csv"),
            },
            "final": {"validation_report": str(raw_dir / "validation_report.yaml")},
        },
        "training": {"id_col": "SK_ID_CURR", "target_col": "TARGET"},
    }


def touch_raw_files(tmp_dir):
    for path in config(tmp_dir)["data"]["raw"].values():
        Path(path).write_text("x\n", encoding="utf-8")


class DataValidationTest(unittest.TestCase):
    def test_raw_path_validation_catches_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            touch_raw_files(tmp_dir)
            Path(config(tmp_dir)["data"]["raw"]["bureau"]).unlink()

            with self.assertRaises(ValueError):
                validate_raw_paths(config(tmp_dir))

    def test_raw_validation_catches_missing_columns_and_duplicate_ids(self):
        train = pl.DataFrame({"SK_ID_CURR": [1, 1], "TARGET": [0, 1]})
        test = pl.DataFrame({"SK_ID_CURR": [2]})
        aux = {
            "bureau": pl.LazyFrame({"SK_ID_CURR": [1]}),
            "bureau_balance": pl.LazyFrame({"SK_ID_BUREAU": [10]}),
            "previous_application": pl.LazyFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [20]}),
            "pos_cash_balance": pl.LazyFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [20]}),
            "installments_payments": pl.LazyFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [20]}),
            "credit_card_balance": pl.LazyFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [20]}),
        }

        with self.assertRaises(ValueError) as error:
            validate_raw_frames(
                train,
                test,
                aux["bureau"],
                aux["bureau_balance"],
                aux["previous_application"],
                aux["pos_cash_balance"],
                aux["installments_payments"],
                aux["credit_card_balance"],
                config(),
            )

        self.assertIn("unique_id.application_train", str(error.exception))
        self.assertIn("required_columns.bureau", str(error.exception))

    def test_final_validation_catches_non_binary_target_and_alignment(self):
        train = pl.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 2], "A": [1.0, 2.0]})
        test = pl.DataFrame({"SK_ID_CURR": [3], "B": [1.0]})

        with self.assertRaises(ValueError) as error:
            validate_final_frames(train, test, config())

        self.assertIn("binary_target.final_train", str(error.exception))
        self.assertIn("feature_alignment.final_test", str(error.exception))

    def test_validation_report_is_written(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = write_validation_report(config(tmp_dir), {"section": [{"name": "ok", "passed": True}]})

            self.assertTrue(path.exists())
            self.assertIn("section", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
