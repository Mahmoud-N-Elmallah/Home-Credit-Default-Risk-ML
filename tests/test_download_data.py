import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from main import validate_step
from src.download_data import (
    COMPETITION_SLUG,
    EXPECTED_RAW_FILENAMES,
    RawDataDownloadError,
    download_raw_data,
    extract_expected_files,
    missing_raw_files,
)


def raw_config(raw_dir):
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
            }
        }
    }


class FakeKaggleApi:
    def __init__(self, archive_path=None, fail_download=False):
        self.archive_path = archive_path
        self.fail_download = fail_download
        self.authenticated = False

    def authenticate(self):
        self.authenticated = True

    def competition_download_files(self, competition, path, quiet=False):
        if self.fail_download:
            raise RuntimeError("download failed")
        if competition != COMPETITION_SLUG:
            raise RuntimeError("wrong competition")
        target = Path(path) / f"{COMPETITION_SLUG}.zip"
        target.write_bytes(Path(self.archive_path).read_bytes())


class DownloadDataTest(unittest.TestCase):
    def test_missing_raw_files_returns_empty_when_all_expected_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir)
            for filename in EXPECTED_RAW_FILENAMES:
                (raw_dir / filename).write_text("x", encoding="utf-8")

            self.assertEqual(missing_raw_files(raw_dir), [])

    def test_download_raw_data_noops_when_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir)
            for filename in EXPECTED_RAW_FILENAMES:
                (raw_dir / filename).write_text("x", encoding="utf-8")

            def fail_if_called():
                raise AssertionError("Kaggle API should not be used")

            self.assertEqual(download_raw_data(raw_config(raw_dir), api_factory=fail_if_called), raw_dir)

    def test_extract_expected_files_places_csvs_in_raw_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_path = base / "raw.zip"
            raw_dir = base / "Raw"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("nested/application_train.csv", "train")
                archive.writestr("ignored.txt", "ignored")

            extracted = extract_expected_files(archive_path, raw_dir)

            self.assertEqual(extracted, {"application_train.csv"})
            self.assertEqual((raw_dir / "application_train.csv").read_text(encoding="utf-8"), "train")
            self.assertFalse((raw_dir / "ignored.txt").exists())

    def test_download_raw_data_raises_when_archive_missing_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_path = base / "partial.zip"
            raw_dir = base / "Raw"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("application_train.csv", "train")

            with self.assertRaisesRegex(RawDataDownloadError, "download incomplete"):
                download_raw_data(raw_config(raw_dir), api_factory=lambda: FakeKaggleApi(archive_path))

    def test_download_raw_data_reports_kaggle_download_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "Raw"

            with self.assertRaisesRegex(RawDataDownloadError, "Kaggle download failed"):
                download_raw_data(raw_config(raw_dir), api_factory=lambda: FakeKaggleApi(fail_download=True))

    def test_main_step_validation_accepts_download_and_reports_expected_values(self):
        self.assertEqual(validate_step("download"), "download")
        with self.assertRaisesRegex(ValueError, "download"):
            validate_step("invalid")


if __name__ == "__main__":
    unittest.main()
