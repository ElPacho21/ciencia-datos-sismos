"""Integridad del DAG que corre el detector en varias regiones."""

from pathlib import Path

import pytest
from airflow.dag_processing.dagbag import DagBag

DAG_ID = "detector_replicas_regions"
DAGS_FOLDER = Path(__file__).resolve().parents[2] / "dags"

DEPENDENCIES = {
    "region_confs": set(),
    "run_region": {"region_confs"},
    "combine": {"run_region"},
}


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=DAGS_FOLDER)


@pytest.fixture(scope="module")
def dag(dagbag):
    return dagbag.dags[DAG_ID]


def test_no_import_errors(dagbag):
    assert not dagbag.import_errors, f"import errors: {dagbag.import_errors}"


def test_triggered_manually(dag):
    assert dag.schedule is None


def test_expected_tasks(dag):
    assert set(dag.task_ids) == set(DEPENDENCIES)


@pytest.mark.parametrize("task", sorted(DEPENDENCIES))
def test_dependencies(dag, task):
    assert dag.get_task(task).upstream_task_ids == DEPENDENCIES[task]


def test_method_params_match_the_region_dag(dagbag, dag):
    """Todo lo que se reenvía a cada región tiene que existir allá."""
    region_dag = dagbag.dags["detector_replicas_api"]
    assert set(dag.params) <= set(region_dag.params)
