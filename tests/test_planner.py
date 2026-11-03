"""Unit tests for planner schemas and DAG validation."""

import unittest
import networkx as nx
from agent_risk.planner.schema import PlanDAG, Route, TaskNode
from agent_risk.planner.api_planner import APIPlanner


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.t1 = TaskNode(
            id="task-1",
            name="read_file",
            description="Read sales data file",
            action_type="file_read",
            inputs={"file": "sales.csv"},
            expected_output="data rows",
            dependencies=[],
            is_critical=True,
        )
        self.t2 = TaskNode(
            id="task-2",
            name="process_statistics",
            description="Compute summary statistics",
            action_type="computation",
            inputs={"ref": "task-1.output"},
            expected_output="statistics dict",
            dependencies=["task-1"],
            is_critical=True,
        )
        self.t3 = TaskNode(
            id="task-3",
            name="send_alert",
            description="Send urgent email to administrator immediately",
            action_type="network_egress",
            inputs={"ref": "task-2.output"},
            expected_output="status 200",
            dependencies=["task-2"],
            is_critical=False,
        )

        self.route = Route(
            route_id="primary-route",
            rationale="Standard 3-step test route",
            tasks=[self.t1, self.t2, self.t3],
        )

        self.plan = PlanDAG(
            plan_id="plan-test-01",
            goal="Read file, calculate stats, and alert admin",
            routes=[self.route],
        )

    def test_plan_dag_validation(self):
        self.assertEqual(self.plan.plan_id, "plan-test-01")
        primary = self.plan.primary_route()
        self.assertEqual(len(primary.tasks), 3)
        self.assertTrue(primary.is_acyclic())

    def test_networkx_conversion(self):
        graph = self.plan.to_networkx()
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))

    def test_cycle_detection(self):
        t1 = TaskNode(id="t1", name="step1", description="", dependencies=["t2"])
        t2 = TaskNode(id="t2", name="step2", description="", dependencies=["t1"])
        route = Route(route_id="cyclic", rationale="", tasks=[t1, t2])
        self.assertFalse(route.is_acyclic())

    def test_api_planner_mock_fallback(self):
        planner = APIPlanner(api_key=None)
        plan = planner.generate_plan("Read customer records and generate pdf report")
        self.assertIsInstance(plan, PlanDAG)
        self.assertGreaterEqual(len(plan.routes), 1)
        self.assertTrue(plan.primary_route().is_acyclic())


if __name__ == "__main__":
    unittest.main()
