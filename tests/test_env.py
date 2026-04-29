import unittest
from unittest.mock import patch

from src.common.env import PROJECT_ROOT, load_project_dotenv


class EnvTest(unittest.TestCase):
    def test_load_project_dotenv_uses_root_env_without_override(self):
        with patch("src.common.env.load_dotenv", return_value=True) as load_dotenv:
            self.assertTrue(load_project_dotenv())

        load_dotenv.assert_called_once_with(PROJECT_ROOT / ".env", override=False)


if __name__ == "__main__":
    unittest.main()
