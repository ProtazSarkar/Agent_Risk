"""Unit tests for configuration."""

import unittest
from agent_risk.config import settings


class TestConfig(unittest.TestCase):
    def test_settings_initialization(self):
        self.assertIn(settings.agent_risk_env, ["development", "testing", "production"])
        self.assertEqual(settings.random_seed, 42)
        self.assertLess(settings.risk_threshold_warning, settings.risk_threshold_block)
        self.assertTrue(settings.data_dir.exists())


if __name__ == "__main__":
    unittest.main()
