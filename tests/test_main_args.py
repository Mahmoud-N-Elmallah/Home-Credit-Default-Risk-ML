import unittest

from main import rewrite_legacy_args


class RewriteLegacyArgsTest(unittest.TestCase):
    def test_rewrites_process_and_config_args(self):
        argv = ["main.py", "--config", "conf/config.yaml", "--process"]

        rewritten = rewrite_legacy_args(argv)

        self.assertEqual(
            rewritten,
            [
                "main.py",
                "--config-path",
                "conf",
                "--config-name",
                "config",
                "run.step=process",
            ],
        )

    def test_all_wins_when_multiple_legacy_steps_are_used(self):
        argv = ["main.py", "--process", "--train"]

        rewritten = rewrite_legacy_args(argv)

        self.assertEqual(rewritten, ["main.py", "run.step=all"])


if __name__ == "__main__":
    unittest.main()
