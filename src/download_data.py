import logging
import shutil
from pathlib import Path
from zipfile import ZipFile

from src.common.env import load_project_dotenv


logger = logging.getLogger(__name__)

COMPETITION_SLUG = "home-credit-default-risk"
EXPECTED_RAW_FILENAMES = {
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
    "sample_submission.csv",
    "HomeCredit_columns_description.csv",
}


class RawDataDownloadError(RuntimeError):
    pass


def raw_data_dir(config):
    raw_paths = [Path(path) for path in config["data"]["raw"].values()]
    if not raw_paths:
        raise RawDataDownloadError("No raw data paths are configured.")
    raw_dirs = {path.parent for path in raw_paths}
    if len(raw_dirs) != 1:
        raise RawDataDownloadError(f"Raw data paths must share one directory. Found: {sorted(map(str, raw_dirs))}")
    return raw_paths[0].parent


def missing_raw_files(raw_dir):
    missing = []
    for filename in sorted(EXPECTED_RAW_FILENAMES):
        path = raw_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def extract_expected_files(archive_path, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            filename = Path(member.filename).name
            if member.is_dir() or filename not in EXPECTED_RAW_FILENAMES:
                continue
            target = raw_dir / filename
            temp_target = target.with_name(f"{target.name}.tmp")
            try:
                with archive.open(member) as source, open(temp_target, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                temp_target.replace(target)
            except Exception:
                temp_target.unlink(missing_ok=True)
                raise
            extracted.add(filename)
    return extracted


def authenticate_kaggle(api_factory=None):
    load_project_dotenv()
    if api_factory is None:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as error:
            raise RawDataDownloadError("Kaggle package is not installed. Run `uv sync`.") from error
        api_factory = KaggleApi

    api = api_factory()
    try:
        api.authenticate()
    except Exception as error:
        raise RawDataDownloadError(
            "Kaggle authentication failed. Configure ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY, "
            "and accept the Home Credit competition rules on Kaggle."
        ) from error
    return api


def download_competition_archive(raw_dir, api_factory=None):
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / f"{COMPETITION_SLUG}.zip"
    archive_path.unlink(missing_ok=True)
    api = authenticate_kaggle(api_factory)
    try:
        api.competition_download_files(COMPETITION_SLUG, path=str(raw_dir), quiet=False)
    except Exception as error:
        raise RawDataDownloadError(
            "Kaggle download failed. Check network access, Kaggle credentials, and competition terms acceptance."
        ) from error
    if not archive_path.exists():
        raise RawDataDownloadError(f"Kaggle download did not create expected archive: {archive_path}")
    return archive_path


def download_raw_data(config, api_factory=None):
    raw_dir = raw_data_dir(config)
    missing = missing_raw_files(raw_dir)
    if not missing:
        logger.info("Raw Kaggle data already present in %s", raw_dir)
        return raw_dir

    logger.info("Missing raw Kaggle files: %s", ", ".join(missing))
    archive_path = download_competition_archive(raw_dir, api_factory=api_factory)
    try:
        extracted = extract_expected_files(archive_path, raw_dir)
    except Exception as error:
        raise RawDataDownloadError("Kaggle archive extraction failed. Remove the downloaded zip and retry.") from error
    archive_path.unlink(missing_ok=True)
    logger.info("Extracted %s Kaggle raw files into %s", len(extracted), raw_dir)

    remaining = missing_raw_files(raw_dir)
    if remaining:
        raise RawDataDownloadError(f"Raw data download incomplete. Missing files: {', '.join(remaining)}")
    return raw_dir
