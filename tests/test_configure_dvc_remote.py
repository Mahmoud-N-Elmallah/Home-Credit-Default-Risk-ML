import unittest
from unittest.mock import ANY, call, patch

from src.common.configure_dvc_remote import configure_dvc_remote, read_dvc_remote_env


class ConfigureDvcRemoteTest(unittest.TestCase):
    def test_read_dvc_remote_env_rejects_missing_values(self):
        with (
            patch("src.common.configure_dvc_remote.load_project_dotenv"),
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "DVC_REMOTE_URL"),
        ):
            read_dvc_remote_env()

    def test_configure_dvc_remote_writes_local_dvc_settings(self):
        values = {
            "DVC_REMOTE_URL": "https://example.test/repo.dvc",
            "DVC_REMOTE_USER": "user",
            "DVC_REMOTE_PASSWORD": "token",
        }

        with patch("src.common.configure_dvc_remote.subprocess.run") as run:
            configure_dvc_remote(values)

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        ANY,
                        "-m",
                        "dvc",
                        "remote",
                        "modify",
                        "--local",
                        "origin",
                        "url",
                        "https://example.test/repo.dvc",
                    ],
                    check=True,
                ),
                call(
                    [ANY, "-m", "dvc", "remote", "modify", "--local", "origin", "auth", "basic"],
                    check=True,
                ),
                call(
                    [ANY, "-m", "dvc", "remote", "modify", "--local", "origin", "user", "user"],
                    check=True,
                ),
                call(
                    [ANY, "-m", "dvc", "remote", "modify", "--local", "origin", "password", "token"],
                    check=True,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
