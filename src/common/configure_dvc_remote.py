import os
import subprocess
import sys

from src.common.env import load_project_dotenv


REQUIRED_DVC_ENV_VARS = ("DVC_REMOTE_URL", "DVC_REMOTE_USER", "DVC_REMOTE_PASSWORD")


def is_placeholder(value):
    return not value or value.startswith("<") or value.startswith("replace_with_")


def read_dvc_remote_env():
    load_project_dotenv()
    values = {name: os.getenv(name, "") for name in REQUIRED_DVC_ENV_VARS}
    missing = [name for name, value in values.items() if is_placeholder(value)]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing DVC environment values: {names}")
    return values


def configure_dvc_remote(values):
    commands = [
        ["remote", "modify", "--local", "origin", "url", values["DVC_REMOTE_URL"]],
        ["remote", "modify", "--local", "origin", "auth", "basic"],
        ["remote", "modify", "--local", "origin", "user", values["DVC_REMOTE_USER"]],
        ["remote", "modify", "--local", "origin", "password", values["DVC_REMOTE_PASSWORD"]],
    ]
    for args in commands:
        subprocess.run([sys.executable, "-m", "dvc", *args], check=True)


def main():
    try:
        values = read_dvc_remote_env()
        configure_dvc_remote(values)
    except Exception as error:
        raise SystemExit(str(error)) from error
    print("DVC local remote config updated in .dvc/config.local")


if __name__ == "__main__":
    main()
