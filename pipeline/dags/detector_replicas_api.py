"""Detector de réplicas sísmicas a partir del servicio FDSN del USGS.

Baja los eventos del rango de fechas pedido a la capa bronze (CSV crudo,
comprimido y particionado por parámetros de consulta) y después refina a
silver. Se dispara a mano porque los parámetros de la ventana los elige quien
lo corre.

EN CONSTRUCCIÓN: ya están Mc, b, d, el bosque de padres, el umbral eta y el
thinning probabilístico; falta armar los clusters y contar las réplicas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

# Se importan los módulos enteros porque varias tareas del DAG se llaman igual
# que la función que las hace, y de otro modo una taparía a la otra.
from sismos import replicas, vecinos
from sismos.bronze import bronze_load, bronze_path, bronze_write
from sismos.parametros import (
    estimate,
    parametros_path,
    parametros_read,
    parametros_write,
)
from sismos.silver import refine, silver_path, silver_read, silver_write
from sismos.umbral import fit_threshold, umbral_path, umbral_read, umbral_write
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
        "seed": Param(
            0,
            type="integer",
            title="Semilla",
            description=(
                "El thinning es probabilístico: baraja el catálogo y sortea qué "
                "enlaces devuelve al fondo. Sin fijar la semilla los resultados "
                "no se reproducen, así que queda registrada en el nombre de los "
                "archivos que produce."
            ),
            minimum=0,
        ),
        "n_randomizaciones": Param(
            5,
            type="integer",
            title="Barajadas del catálogo nulo",
            description=(
                "Cuántas veces se baraja el catálogo para estimar la "
                "distribución de eta bajo puro azar. Más barajadas dan un nulo "
                "menos ruidoso, pero cada una es una corrida completa del vecino "
                "más cercano."
            ),
            minimum=1,
            maximum=50,
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

    @task
    def fit_eta_threshold(vecinos_ruta: str, **context) -> str:
        params = context["params"]

        destino = umbral_path(
            starttime=params["starttime"],
            endtime=params["endtime"],
            minmagnitude=params["minmagnitude"],
        )
        fuente = Path(vecinos_ruta)

        if (
            destino.exists()
            and not params["force"]
            and destino.stat().st_mtime >= fuente.stat().st_mtime
        ):
            log.info("Se reutilizó un umbral ya ajustado.")
            return str(destino)

        umbral = fit_threshold(vecinos.vecinos_read(fuente))
        umbral_write(destino, umbral)
        return str(destino)

    @task
    def randomize_catalog(silver_ruta: str, parametros_ruta: str, **context) -> str:
        params = context["params"]

        destino = replicas.nulo_path(
            starttime=params["starttime"],
            endtime=params["endtime"],
            minmagnitude=params["minmagnitude"],
            seed=params["seed"],
            n_repeticiones=params["n_randomizaciones"],
        )
        fuente = Path(parametros_ruta)

        # La semilla y las repeticiones ya están en el nombre del archivo, así
        # que alcanza con comparar contra los parámetros de aguas arriba.
        if (
            destino.exists()
            and not params["force"]
            and destino.stat().st_mtime >= fuente.stat().st_mtime
        ):
            log.info("Se reutilizó un catálogo nulo ya persistido.")
            return str(destino)

        parametros = parametros_read(fuente)
        nulo = replicas.randomize_catalog(
            silver_read(Path(silver_ruta)),
            b=parametros["b"],
            d=parametros["d"],
            mc=parametros["mc"],
            n_repeticiones=params["n_randomizaciones"],
            seed=params["seed"],
        )
        replicas.nulo_write(destino, nulo)
        return str(destino)

    @task
    def thinning(
        vecinos_ruta: str, nulo_ruta: str, umbral_ruta: str, **context
    ) -> str:
        params = context["params"]

        destino = replicas.replicas_path(
            starttime=params["starttime"],
            endtime=params["endtime"],
            minmagnitude=params["minmagnitude"],
            seed=params["seed"],
        )
        fuentes = [Path(vecinos_ruta), Path(nulo_ruta), Path(umbral_ruta)]

        if (
            destino.exists()
            and not params["force"]
            and all(destino.stat().st_mtime >= f.stat().st_mtime for f in fuentes)
        ):
            log.info("Se reutilizó una clasificación de réplicas ya persistida.")
            return str(destino)

        umbral = umbral_read(Path(umbral_ruta))
        emparentados = vecinos.vecinos_read(Path(vecinos_ruta))

        # El nulo no entra en la clasificación: sirve para saber si la
        # bimodalidad que se está usando para clasificar es real.
        contraste = replicas.contrastar_con_nulo(
            emparentados,
            replicas.nulo_read(Path(nulo_ruta)),
            log10_eta0=umbral["log10_eta0"],
        )

        # El sorteo usa una semilla derivada de la del DAG: barajar el catálogo
        # y sortear los enlaces son cosas distintas y no comparten el flujo.
        clasificados, diagnostico = replicas.thin(
            emparentados,
            mezcla=umbral["mezcla"],
            log10_eta0=umbral["log10_eta0"],
            seed=params["seed"] + 1,
        )
        replicas.replicas_write(destino, clasificados)
        log.info("Thinning: %s | contraste con el nulo: %s", diagnostico, contraste)
        return str(destino)

    silver_ruta = refine_silver(land_bronze())
    parametros_ruta = estimate_mc_b_d(silver_ruta)
    vecinos_ruta = nearest_neighbor(silver_ruta, parametros_ruta)

    thinning(
        vecinos_ruta,
        randomize_catalog(silver_ruta, parametros_ruta),
        fit_eta_threshold(vecinos_ruta),
    )


# Sin esta llamada el DAG no queda registrado: el decorador @dag sólo devuelve
# una factory, es invocarla lo que lo publica en el DagBag.
detector_replicas_api()
