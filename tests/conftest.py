"""Pytest shared fixtures."""

import pytest
from agent_risk.planner.schema import PlanDAG, Route, TaskNode


@pytest.fixture
def sample_task_node() -> TaskNode:
    return TaskNode(
        id="task-1",
        name="read_secure_file",
        description="Read file safely from local repository",
        action_type="file_read",
        inputs={"path": "sample.txt"},
        expected_output="file content",
        dependencies=[],
        is_critical=True,
    )


@pytest.fixture
def sample_plan_dag() -> PlanDAG:
    t1 = TaskNode(
        id="task-1",
        name="read_file",
        description="Read sales data file",
        action_type="file_read",
        inputs={"file": "sales.csv"},
        expected_output="data rows",
        dependencies=[],
        is_critical=True,
    )
    t2 = TaskNode(
        id="task-2",
        name="process_statistics",
        description="Compute summary statistics",
        action_type="computation",
        inputs={"ref": "task-1.output"},
        expected_output="statistics dict",
        dependencies=["task-1"],
        is_critical=True,
    )
    t3 = TaskNode(
        id="task-3",
        name="send_alert",
        description="Send urgent email to administrator immediately",
        action_type="network_egress",
        inputs={"ref": "task-2.output"},
        expected_output="status 200",
        dependencies=["task-2"],
        is_critical=False,
    )

    route = Route(
        route_id="primary-route",
        rationale="Standard 3-step test route",
        tasks=[t1, t2, t3],
    )

    return PlanDAG(
        plan_id="plan-test-01",
        goal="Read file, calculate stats, and alert admin",
        routes=[route],
    )

