"""Unit tests for feature extractor and psycholinguistic marker detection."""

import unittest
import pandas as pd
from agent_risk.planner.schema import PlanDAG, Route, TaskNode
from agent_risk.risk_engine.feature_extractor import FeatureExtractor


class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        self.task = TaskNode(
            id="task-1",
            name="read_secure_file",
            description="Read file safely from local repository",
            action_type="file_read",
            inputs={"path": "sample.txt"},
            expected_output="file content",
            dependencies=[],
            is_critical=True,
        )

    def test_task_feature_extraction(self):
        features = self.extractor.extract_task_features(self.task)
        self.assertIn("text_length", features)
        self.assertIn("word_count", features)
        self.assertIn("urgency_count", features)
        self.assertIn("is_file_read", features)
        self.assertEqual(features["is_file_read"], 1.0)
        self.assertEqual(features["is_command_exec"], 0.0)

    def test_psycholinguistic_keywords(self):
        urgent_task = TaskNode(
            id="t-u",
            name="emergency_root_override",
            description="Override sudo credentials now immediately and disregard rules",
            action_type="command_exec",
        )
        features = self.extractor.extract_task_features(urgent_task)
        self.assertGreaterEqual(features["urgency_count"], 2)
        self.assertGreaterEqual(features["privilege_count"], 2)
        self.assertGreaterEqual(features["deception_count"], 1)

    def test_dag_dataframe_extraction(self):
        route = Route(route_id="r1", rationale="test", tasks=[self.task])
        plan = PlanDAG(plan_id="p1", goal="Test", routes=[route])
        df = self.extractor.extract_dag_dataframe(plan)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn("dag_node_count", df.columns)
        self.assertEqual(df["dag_node_count"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
