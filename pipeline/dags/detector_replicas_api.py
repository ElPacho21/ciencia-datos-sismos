"""Detector de réplicas sísmicas a partir del servicio FDSN del USGS.

Baja los eventos del rango de fechas pedido a la capa bronze (CSV crudo,
comprimido y particionado por parámetros de consulta) y después refina a
silver. Se dispara a mano porque los parámetros de la ventana los elige quien
lo corre.

EN CONSTRUCCIÓN: ya están Mc, b, d y el bosque de padres; falta el umbral
eta, el thinning y el armado de los clusters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

# Se importa el módulo entero porque la tarea del DAG se llama igual que la
# función que la hace, y de otro modo una taparía a la otra.
from sismos import vecinos
from sismos.bronze import bronze_load, bronze_path, bronze_write
from sismos.parametros import (
    estimate,
    parametros_path,
    parametros_read,
    parametros_write,
)
from sismos.silver import refine, silver_path, silver_read, silver_write
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
        "mc_metodo": Param(
            "gft",
            type="string",
            enum=["gft", "maxc"],
            title="Método para la magnitud de completitud",
            description=(
                "Cómo estimar Mc. 'gft' es bondad de ajuste (Wiemer & Wyss), "
                "más exigente; 'maxc' es máxima curvatura, más permisivo y "
                "usado como fallback cuando gft no llega al objetivo."
            ),
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

    @task
    def estimate_mc_b_d(silver_ruta: str, **context) -> str:
        params = context["params"]

        destino = parametros_path(
            starttime=params["starttime"],
            endtime=params["endtime"],
            minmagnitude=params["minmagnitude"],
        )

        if destino.exists() and not params["force"]:
            guardado = parametros_read(destino)
            # A diferencia de bronze y silver, acá la partición no alcanza para
            # decidir si sirve lo persistido: los mismos datos con otro método
            # de Mc dan otros parámetros. Se comparan los knobs con los que se
            # calculó y sólo se reutiliza si son los mismos.
            if guardado.get("knobs", {}).get("mc_metodo") == params["mc_metodo"]:
                log.info(
                    "Se reutilizaron los parámetros ya estimados: Mc=%.2f, b=%.3f, d=%.3f.",
                    guardado["mc"],
                    guardado["b"],
                    guardado["d"],
                )
                return str(destino)
            log.info(
                "Los parámetros persistidos son de otro método de Mc: se recalcula."
            )

        parametros = estimate(
            silver_read(Path(silver_ruta)), mc_metodo=params["mc_metodo"]
        )
        parametros_write(destino, parametros)
        return str(destino)

    @task
    def nearest_neighbor(silver_ruta: str, parametros_ruta: str, **context) -> str:
        params = context["params"]

        destino = vecinos.vecinos_path(
            starttime=params["starttime"],
            endtime=params["endtime"],
            minmagnitude=params["minmagnitude"],
        )
        fuente = Path(parametros_ruta)

        # Acá no alcanza con que el archivo exista: el bosque de padres depende
        # de Mc, b y d, que se recalculan aguas arriba. Si el json de
        # parámetros es más nuevo que este parquet, lo que hay quedó viejo.
        if (
            destino.exists()
            and not params["force"]
            and destino.stat().st_mtime >= fuente.stat().st_mtime
        ):
            log.info("Se reutilizó un bosque de padres ya persistido.")
            return str(destino)

        parametros = parametros_read(fuente)
        emparentados = vecinos.nearest_neighbor(
            silver_read(Path(silver_ruta)),
            b=parametros["b"],
            d=parametros["d"],
            mc=parametros["mc"],
        )
        vecinos.vecinos_write(destino, emparentados)
        return str(destino)

    silver_ruta = refine_silver(land_bronze())
    nearest_neighbor(silver_ruta, estimate_mc_b_d(silver_ruta))


# Sin esta llamada el DAG no queda registrado: el decorador @dag sólo devuelve
# una factory, es invocarla lo que lo publica en el DagBag.
detector_replicas_api()
