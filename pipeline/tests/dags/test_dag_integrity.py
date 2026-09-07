"""Chequeos de integridad del DAG: que parsee, que quede registrado y que el
grafo de tareas sea el que se espera.

El grafo importa más de lo que parece. Cada paso del método consume lo que
produjo el anterior y varios dependen de *dos* insumos a la vez, así que un
enlace de menos no rompe el DAG: lo hace correr con datos viejos o en el orden
equivocado. Estos tests fijan esa forma.

Sobre la API que se usa acá: en Airflow 3 el `DagBag` que parsea archivos vive
en `airflow.dag_processing.dagbag` y ya no acepta `include_examples` — se le
pasa directamente la carpeta a leer. Y se accede a `dagbag.dags[...]` en vez de
`get_dag()`, que consulta la base: para chequear que un archivo parsea no hace
falta una base de datos.
"""

from pathlib import Path

import pytest
from airflow.dag_processing.dagbag import DagBag

DAG_ID = "detector_replicas_api"
CARPETA_DAGS = Path(__file__).resolve().parents[2] / "dags"

# Quién necesita a quién. Es el contrato del pipeline escrito una sola vez.
DEPENDENCIAS = {
    "land_bronze": set(),
    "refine_silver": {"land_bronze"},
    "estimate_mc": {"refine_silver"},
    # El vecino más cercano necesita el catálogo y además Mc, b y d.
    "nearest_neighbor": {"refine_silver", "estimate_mc"},
    "fit_eta_threshold": {"nearest_neighbor"},
    "randomize_catalog": {"refine_silver", "estimate_mc"},
    # El thinning cruza el bosque de padres, el umbral y el catálogo nulo.
    "thinning": {"nearest_neighbor", "randomize_catalog", "fit_eta_threshold"},
    "build_clusters": {"thinning"},
    # La entrega es el último paso y no alimenta a nadie: escribe el csv y lo
    # valida.
    "publish_csv": {"build_clusters"},
}


@pytest.fixture(scope="module")
def dagbag():
    """Parsear la carpeta es caro: se hace una sola vez para todo el módulo."""
    return DagBag(dag_folder=CARPETA_DAGS)


@pytest.fixture(scope="module")
def dag(dagbag):
    return dagbag.dags[DAG_ID]


def test_no_import_errors(dagbag):
    assert not dagbag.import_errors, f"errores de import: {dagbag.import_errors}"


def test_dag_registrado(dagbag):
    assert DAG_ID in dagbag.dags, f"sólo se encontraron {dagbag.dag_ids}"


def test_dag_se_dispara_a_mano(dag):
    """El DAG no corre solo: la ventana la elige quien lo dispara."""
    assert dag.schedule is None


def test_params_de_la_consulta(dag):
    """Los parámetros de la consulta son la interfaz contra la API FDSN."""
    assert {
        "starttime",
        "endtime",
        "limit",
        "minmagnitude",
        "minlatitude",
        "maxlatitude",
        "minlongitude",
        "maxlongitude",
    } <= set(dag.params)


def test_params_del_metodo(dag):
    """Las decisiones de método también son parámetros, no constantes."""
    assert {"mainshock", "seed", "n_randomizaciones"} <= set(dag.params)


def test_tareas_esperadas(dag):
    assert set(dag.task_ids) == set(DEPENDENCIAS)


@pytest.mark.parametrize("tarea", sorted(DEPENDENCIAS))
def test_dependencias(dag, tarea):
    assert dag.get_task(tarea).upstream_task_ids == DEPENDENCIAS[tarea]


def test_el_pipeline_termina_en_la_entrega(dag):
    """Ninguna tarea queda colgada sin alimentar a nadie salvo la última."""
    hojas = sorted(t.task_id for t in dag.tasks if not t.downstream_task_ids)
    assert hojas == ["publish_csv"], f"el pipeline termina en {hojas}"
