"""Detector de réplicas sísmicas a partir del servicio FDSN del USGS.

Baja los eventos del rango de fechas pedido a la capa bronze (CSV crudo,
comprimido y particionado por parámetros de consulta) y después refina a
silver. Se dispara a mano porque los parámetros de la ventana los elige quien
lo corre.

EN CONSTRUCCIÓN: falta todo lo que es propio del método de Zaliapin &
Ben-Zion (magnitud de completitud, b, d, vecino más cercano, thinning).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task
from sismos.bronze import bronze_load, bronze_path, bronze_write
from sismos.silver import refine, silver_path, silver_write
from sismos.usgs_earthquake import fetch

log = logging.getLogger(__name__)


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
        "minmagnitude": Param(
            0,
            type="number",
            title="Magnitud mínima",
            description="Magnitud mínima de los sismos consultados.",
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
        "force": Param(
            False,
            type="boolean",
            title="Forzar la corrida",
            description=(
                "Ignora todas las cachés: baja aunque la fuente no "
                "haya cambiado, y vuelve a pedir el csv que ya "
                "en bronce."
            ),
        ),
    },
)
def detector_replicas_api():
    # Las rutas viajan entre tareas como str: el XCom se serializa a JSON y un
    # Path no sobrevive el viaje.
    @task
    def land_bronze(**context) -> str:
        params = context["params"]

        starttime = params["starttime"]
        endtime = params["endtime"]
        minmagnitude = params["minmagnitude"]

        destino = bronze_path(
            starttime=starttime, endtime=endtime, minmagnitude=minmagnitude
        )

        if destino.exists() and not params["force"]:
            log.info("Se reutilizó un csv ya persistido.")
            return str(destino)

        csv = fetch(starttime=starttime, endtime=endtime, minmagnitude=minmagnitude)
        bronze_write(destino, csv)
        log.info("Se bajó el csv de la API a %s.", destino)
        return str(destino)

    @task
    def refine_silver(bronze_ruta: str, **context) -> str:
        params = context["params"]

        destino = silver_path(
            starttime=params["starttime"],
            endtime=params["endtime"],
            minmagnitude=params["minmagnitude"],
        )

        if destino.exists() and not params["force"]:
            log.info("Se reutilizó un parquet de silver ya persistido.")
            return str(destino)

        sismos = refine(bronze_load(Path(bronze_ruta)))
        silver_write(destino, sismos)
        log.info("Se refinaron %d eventos a %s.", len(sismos), destino)
        return str(destino)

    refine_silver(land_bronze())


# Sin esta llamada el DAG no queda registrado: el decorador @dag sólo devuelve
# una factory, es invocarla lo que lo publica en el DagBag.
detector_replicas_api()
