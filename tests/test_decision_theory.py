"""Unit tests for Markov risk chains and Stackelberg shield policies."""

import unittest
import numpy as np
from agent_risk.decision_theory.game_theory import StackelbergShield
from agent_risk.decision_theory.markov_chain import MarkovRiskModel
from agent_risk.planner.schema import Route, TaskNode


class TestDecisionTheory(unittest.TestCase):
    def test_markov_transition_matrix(self):
        model = MarkovRiskModel()
        T = model.build_transition_matrix([0.1, 0.2, 0.3])

        # Check rows sum to 1.0 (valid stochastic matrix)
        row_sums = T.sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0, 1.0], rtol=1e-5)

        # Check absorbing states
        self.assertEqual(T[2, 2], 1.0)  # Compromised is absorbing
        self.assertEqual(T[3, 3], 1.0)  # Goal is absorbing

    def test_markov_trajectory_evaluation(self):
        model = MarkovRiskModel()
        tasks = [
            TaskNode(id="t1", name="step1", description="", risk_score=0.1),
            TaskNode(id="t2", name="step2", description="", risk_score=0.2),
        ]
        route = Route(route_id="r1", rationale="", tasks=tasks)
        res = model.evaluate_route_survival(route, {"t1": 0.1, "t2": 0.2})

        self.assertIn("trajectory_compromise_prob", res)
        self.assertIn("goal_reach_prob", res)
        self.assertTrue(0.0 <= res["trajectory_compromise_prob"] <= 1.0)

    def test_stackelberg_equilibrium_search(self):
        game = StackelbergShield(cost_false_alarm=1.0, cost_security_breach=10.0, reward_task_success=3.0)

        # Low risks benign, high risks adversarial
        risks = [0.05, 0.10, 0.15, 0.85, 0.90, 0.95]
        labels = [0, 0, 0, 1, 1, 1]

        best_th, best_u, curve = game.solve_optimal_threshold(risks, labels)

        self.assertTrue(0.15 <= best_th <= 0.85)
        self.assertGreater(best_u, 0.0)


if __name__ == "__main__":
    unittest.main()
