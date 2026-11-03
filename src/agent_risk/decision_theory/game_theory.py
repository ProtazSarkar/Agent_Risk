"""Game-Theoretic Shield Policies and Adversarial Equilibria.

Models security monitoring as a Stackelberg Leader-Follower game between a defensive
inspection shield (Leader) and an untrusted/adversarial agent actor (Follower).
"""

from typing import Dict, List, Tuple
import numpy as np


class StackelbergShield:
    """Computes optimal defensive risk thresholds against rational or bounded-rational adversaries."""

    def __init__(
        self,
        cost_false_alarm: float = 1.0,
        cost_security_breach: float = 10.0,
        reward_task_success: float = 3.0,
    ):
        self.c_fa = cost_false_alarm
        self.c_breach = cost_security_breach
        self.r_success = reward_task_success

    def defender_utility(
        self,
        threshold: float,
        is_adversarial: bool,
        predicted_risk: float,
    ) -> float:
        """Utility function for the defender shield:
        - If task is blocked (predicted_risk >= threshold):
            - If it was adversarial: +reward (avoided breach)
            - If it was benign: -cost_false_alarm (operational friction)
        - If task is allowed (predicted_risk < threshold):
            - If it was adversarial: -cost_security_breach (exploit executed)
            - If it was benign: +reward_task_success
        """
        blocked = predicted_risk >= threshold

        if blocked:
            if is_adversarial:
                return 2.0  # Safe defense bonus
            else:
                return -self.c_fa
        else:
            if is_adversarial:
                return -self.c_breach
            else:
                return self.r_success

    def solve_optimal_threshold(
        self,
        sample_risks: List[float],
        sample_labels: List[int],
        candidate_thresholds: List[float] = None,
    ) -> Tuple[float, float, Dict[float, float]]:
        """Search over candidate decision thresholds to maximize defender expected utility (Stackelberg leader).

        Returns:
            (best_threshold, max_expected_utility, utility_curve)
        """
        if candidate_thresholds is None:
            candidate_thresholds = list(np.linspace(0.1, 0.9, 17))

        curve: Dict[float, float] = {}
        best_threshold = 0.5
        best_util = -float("inf")

        for th in candidate_thresholds:
            utils = [
                self.defender_utility(th, bool(label), risk)
                for risk, label in zip(sample_risks, sample_labels)
            ]
            expected_u = float(np.mean(utils)) if utils else 0.0
            curve[round(th, 2)] = round(expected_u, 4)

            if expected_u > best_util:
                best_util = expected_u
                best_threshold = round(th, 2)

        return best_threshold, best_util, curve

