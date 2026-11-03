"""Markov Chain & MDP Trajectory Risk Modeling.

Models sequential agent execution as a state transition system where cumulative risk,
absorbing hazard states, and trajectory survival probabilities are analytically evaluated.
"""

from typing import Dict, List, Tuple
import numpy as np
from agent_risk.planner.schema import Route


class MarkovRiskModel:
    """Evaluates multi-step trajectory risk through state transition probability matrices."""

    def __init__(self, states: List[str] = None):
        # Default states: Safe (S), Warning (W), Absorbing Compromise (C), Terminal Goal (G)
        self.states = states or ["Safe", "Warning", "Compromised", "Goal"]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.n_states = len(self.states)

    def build_transition_matrix(
        self,
        step_risk_probabilities: List[float],
    ) -> np.ndarray:
        """Construct an empirical or parameterized transition matrix for a sequence of steps.

        Transitions:
        - Compromised is an absorbing state.
        - Higher step risk shifts mass from Safe towards Warning and Compromised.
        """
        avg_risk = float(np.mean(step_risk_probabilities)) if step_risk_probabilities else 0.0
        T = np.zeros((self.n_states, self.n_states))

        # Safe state transitions: [Safe, Warning, Compromised, Goal]
        p_c = min(0.95, avg_risk)
        p_w = min(0.95 - p_c, (1.0 - p_c) * 0.4)
        p_s = max(0.0, 1.0 - p_c - p_w - 0.2)
        p_g = max(0.0, 1.0 - (p_c + p_w + p_s))

        T[0, :] = [p_s, p_w, p_c, p_g]

        # Warning state: more likely to transition to Compromised
        T[1, :] = [p_s * 0.5, p_w * 0.5, min(0.98, p_c * 1.5 + 0.1), 0.1]
        T[1, :] /= T[1, :].sum()

        # Compromised is absorbing: P(C -> C) = 1.0
        T[2, 2] = 1.0

        # Goal is absorbing: P(G -> G) = 1.0
        T[3, 3] = 1.0

        return T

    def evaluate_route_survival(
        self,
        route: Route,
        task_risk_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute the cumulative probability of reaching the Goal state without absorbing into Compromised."""
        risks = [task_risk_scores.get(t.id, t.risk_score) for t in route.tasks]
        if not risks:
            return {"goal_reach_prob": 1.0, "compromise_prob": 0.0, "cumulative_hazard": 0.0}

        T = self.build_transition_matrix(risks)

        # Initial state vector: start in Safe state [1, 0, 0, 0]
        v = np.zeros(self.n_states)
        v[0] = 1.0

        # Propagate through horizon equal to number of tasks
        for _ in route.tasks:
            v = np.dot(v, T)

        # Cumulative survival metrics
        prob_compromised = float(v[self.state_to_idx["Compromised"]])
        prob_goal = float(v[self.state_to_idx["Goal"]])

        # Independent cumulative risk failure model: 1 - Prod(1 - r_i)
        indep_failure = 1.0 - float(np.prod([1.0 - min(0.99, r) for r in risks]))

        return {
            "trajectory_compromise_prob": prob_compromised,
            "goal_reach_prob": prob_goal,
            "independent_hazard_score": indep_failure,
            "mean_step_risk": float(np.mean(risks)),
        }

