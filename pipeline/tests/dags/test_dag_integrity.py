"""Chequeos mínimos sobre el DAG: que parsee y que quede registrado.

Todavía no se afirma nada sobre el grafo de tareas porque está a medio armar.
Cuando `detector_replicas_api` tenga sus tareas encadenadas, agregar acá un
test de tareas esperadas como el de airflow-fifa.
"""
from airflow.models import DagBag


def test_no_import_errors():
    dagbag = DagBag(include_examples=False)
    assert not dagbag.import_errors, f"errores de import: {dagbag.import_errors}"


def test_dag_registrado():
    dagbag = DagBag(include_examples=False)
    assert dagbag.get_dag("detector_replicas_api") is not None


def test_dag_se_dispara_a_mano():
    """Mientras se desarrolla, el DAG no corre solo: se dispara con parámetros."""
    dagbag = DagBag(include_examples=False)
    assert dagbag.get_dag("detector_replicas_api").schedule is None


def test_params_de_la_consulta():
    """Los cuatro parámetros son la interfaz del DAG contra la API FDSN."""
    dagbag = DagBag(include_examples=False)
    params = dagbag.get_dag("detector_replicas_api").params
    assert {"starttime", "endtime", "limit", "minmagnitude"} <= set(params)
