"""Detector de réplicas sísmicas a partir del servicio FDSN del USGS.

Baja los eventos del rango de fechas pedido a la capa bronze (CSV crudo,
comprimido y particionado por parámetros de consulta) y después refina a
silver. Se dispara a mano porque los parámetros de la ventana los elige quien
lo corre.

EN CONSTRUCCIÓN: las tareas todavía no están encadenadas.
"""

from __future__ import annotations

import logging

import pendulum
from airflow.sdk import Param, dag, task
from sismos.bronze import (
    bronze_path,
)

log = logging.getLogger(__name__)

CONN_ID = "sismosapi"

EVENT_QUERY = "/fdsnws/event/1/query"
EVENT_COUND = "/fdsnws/event/1/count"


@dag(
    dag_id="detector_replicas_api",
    schedule=None,
    start_date=pendulum.datetime(2026, 8, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["ciencias-de-datos", "proyecto-integrador"],
    doc_md=__doc__,
    params={
        "starttime": Param(
            type="string",
            format="date",
            title="Fecha de inicio",
            description=("Fecha a partir de la cual obtener registros de sismos."),
        ),
        "endtime": Param(
            type="string",
            format="date",
            title="Fecha de fin",
            description=("Fecha hasta la cual obtener registros de sismos."),
        ),
        "limit": Param(
            20000,
            type="integer",
            title="Cantidad de sismos límite",
            description=(
                "Indica el límite de la cantidad de sismos a obtener entre las fechas dadas. De 0 hasta 20000"
            ),
            minimum=0,
            maximum=20000,
        ),
        "minmagnitude": Param(
            0,
            type="number",
            title="Magnitud mínima",
            description="Magnitud mínima de los sismos consultados.",
        ),
    },
)
def detector_replicas_api():
    @task()
    def obtener_sismos_dict(**context):
        params = context["params"]
        return {
            "starttime": params["starttime"],
            "endtime": params["endtime"],
            "minmagnitude": params["minmagnitude"],
        }

    @task()
    def land_bronze(sismos: dict) -> dict:
        destino = bronze_path(
            sismos["starttime"], sismos["endtime"], sismos["minmagnitude"]
        )
        if destino.exists() and not sismos["force"]:
            log.info("Se reutilizó un csv ya persistido.")
        else:
            log.info("Se reutilizó un csv ya persistido.")
        return destino

    @task()
    def refine_silver(sismos: dict) -> dict:
        pass

    sismos_dict = obtener_sismos_dict()
    land_bronze.expand(sismos=sismos_dict)


# Sin esta llamada el DAG no queda registrado: el decorador @dag sólo devuelve
# una factory, es invocarla lo que lo publica en el DagBag.
detector_replicas_api()
