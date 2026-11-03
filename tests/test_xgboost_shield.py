"""Unit tests for XGBoost Risk Shield and Necessity Filter."""

import unittest
import pandas as pd
from agent_risk.planner.schema import Route, TaskNode
from agent_risk.risk_engine.necessity_filter import NecessityFilter
from agent_risk.risk_engine.xgboost_shield import RiskVerdict, XGBoostShield


class TestXGBoostShield(unittest.TestCase):
    def test_xgboost_shield_heuristic_baseline(self):
        shield = XGBoostShield()
        shield.load_model()
        self.assertIsNotNone(shield.model)

        # Test dummy benign features
        benign_x = pd.DataFrame([
            [10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        ], columns=shield.feature_extractor.feature_names)

        probs = shield.predict_proba(benign_x)
        self.assertEqual(len(probs), 1)
        self.assertTrue(0.0 <= probs[0] <= 1.0)

        verdict, score = shield.evaluate_verdict(probs[0])
        self.assertIn(verdict, [RiskVerdict.SAFE, RiskVerdict.WARNING, RiskVerdict.BLOCKED])

    def test_necessity_filter_detects_dead_branches(self):
        filter_engine = NecessityFilter(necessity_threshold=0.40)

        t1 = TaskNode(id="t1", name="init", description="", is_critical=True)
        t2 = TaskNode(id="t2", name="process", description="", dependencies=["t1"], is_critical=True)
        # Disconnected dead-end task
        t_dead = TaskNode(id="t_dead", name="useless_scan", description="", dependencies=["t1"], is_critical=False)
        t3 = TaskNode(id="t3", name="output", description="", dependencies=["t2"], is_critical=True)

        route = Route(route_id="r1", rationale="", tasks=[t1, t2, t_dead, t3])
        scores = filter_engine.score_route(route)

        self.assertGreater(scores["t1"], scores["t_dead"])
        self.assertGreater(scores["t3"], scores["t_dead"])
        self.assertLess(scores["t_dead"], 0.40)

        necessary, pruned = filter_engine.filter_unnecessary_tasks(route)
        self.assertEqual(len(necessary), 3)
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].id, "t_dead")


if __name__ == "__main__":
    unittest.main()
