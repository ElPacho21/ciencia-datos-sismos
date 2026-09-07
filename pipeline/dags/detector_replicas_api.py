"""Detector de réplicas sísmicas a partir del servicio FDSN del USGS.

Baja los eventos del rango de fechas pedido a la capa bronze (CSV crudo,
comprimido y particionado por parámetros de consulta) y después refina a
silver. Se dispara a mano porque los parámetros de la ventana los elige quien
lo corre.

De ahí en adelante corre el método de Zaliapin & Ben-Zion: estima Mc, tiende el
bosque de padres por vecino más cercano, ajusta el umbral eta contra un
catálogo barajado, sortea las réplicas por thinning y arma los clusters. La
última tarea publica el csv y lo valida.

Una fila del entregable es **una secuencia sísmica**: un sismo principal con
todas sus réplicas. La clave es `cluster_id` y la columna objetivo,
`n_replicas`. Se publica además el catálogo evento por evento, donde una fila
es un terremoto y el objetivo es `is_aftershock`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

# Se importan los módulos enteros porque varias tareas del DAG se llaman igual
# que la función que las hace, y de otro modo una taparía a la otra.
from sismos import clusters, entrega, replicas, vecinos
from sismos.bronze import bronze_load, bronze_path, bronze_write
from sismos.parametros import estimate
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
        # Rectángulo por defecto: Argentina continental. El método supone un
        # catálogo homogéneo, y el mundo entero no lo es — mezcla regiones con
        # completitudes muy distintas y hace que los árboles de réplicas
        # encadenen zonas sin relación. Dejar los cuatro en null vuelve a
        # consultar el planeta completo.
        "minlatitude": Param(
            -55,
            type=["number", "null"],
            title="Latitud mínima",
            description="Borde sur del rectángulo a consultar.",
            minimum=-90,
            maximum=90,
        ),
        "maxlatitude": Param(
            -21,
            type=["number", "null"],
            title="Latitud máxima",
            description="Borde norte del rectángulo a consultar.",
            minimum=-90,
            maximum=90,
        ),
        "minlongitude": Param(
            -74,
            type=["number", "null"],
            title="Longitud mínima",
            description="Borde oeste del rectángulo a consultar.",
            minimum=-180,
            maximum=180,
        ),
        "maxlongitude": Param(
            -53,
            type=["number", "null"],
            title="Longitud máxima",
            description="Borde este del rectángulo a consultar.",
            minimum=-180,
            maximum=180,
        ),
        "limit": Param(
            20000,
            type="integer",
            title="Tope de sismos a bajar",
            description=(
                "Tope de seguridad, no un recorte: si el rango empareja más "
                "sismos que esto, la corrida falla en vez de bajar un catálogo "
                "truncado. 0 significa sin tope — se baja la ventana completa, "
                "partiéndola en varios pedidos si supera los 20000 que admite "
                "el servicio por consulta."
            ),
            minimum=0,
            maximum=20000,
        ),
        "mainshock": Param(
            "mayor",
            type="string",
            enum=["mayor", "raiz"],
            title="Definición de sismo principal",
            description=(
                "Cuál de los eventos del cluster es el sismo principal. "
                "'mayor' es el de mayor magnitud, que es lo que usan Zaliapin & "
                "Ben-Zion; 'raiz' es el que disparó la secuencia. Difieren "
                "cuando hubo premonitores."
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
    def consulta(params) -> dict:
        return {
            clave: params[clave]
            for clave in (
                "starttime",
                "endtime",
                "minmagnitude",
                "minlatitude",
                "maxlatitude",
                "minlongitude",
                "maxlongitude",
            )
        }

    @task
    def land_bronze(**context) -> str:
        params = context["params"]

        destino = bronze_path(**consulta(params))

        if destino.exists() and not params["force"]:
            log.info("Se reutilizó un csv ya persistido.")
            return str(destino)

        csv = fetch(limit=params["limit"], **consulta(params))
        bronze_write(destino, csv)
        log.info("Se bajó el csv de la API a %s.", destino)
        return str(destino)

    @task
    def refine_silver(bronze_ruta: str, **context) -> str:
        params = context["params"]

        destino = silver_path(**consulta(params))

        if destino.exists() and not params["force"]:
            log.info("Se reutilizó un parquet de silver ya persistido.")
            return str(destino)

        sismos = refine(bronze_load(Path(bronze_ruta)))
        silver_write(destino, sismos)
        log.info("Se refinaron %d eventos a %s.", len(sismos), destino)
        return str(destino)

    @task
    def estimate_mc(silver_ruta: str) -> dict:
        """Mc del catálogo, más los b y d estándar."""
        return estimate(silver_read(Path(silver_ruta)))

    @task
    def nearest_neighbor(silver_ruta: str, parametros: dict, **context) -> str:
        params = context["params"]

        destino = vecinos.vecinos_path(**consulta(params))
        fuente = Path(silver_ruta)

        if (
            destino.exists()
            and not params["force"]
            and destino.stat().st_mtime >= fuente.stat().st_mtime
        ):
            log.info("Se reutilizó un bosque de padres ya persistido.")
            return str(destino)

        emparentados = vecinos.nearest_neighbor(
            silver_read(fuente),
            b=parametros["b"],
            d=parametros["d"],
            mc=parametros["mc"],
        )
        vecinos.vecinos_write(destino, emparentados)
        return str(destino)

    @task
    def fit_eta_threshold(vecinos_ruta: str, **context) -> str:
        params = context["params"]

        destino = umbral_path(**consulta(params))
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
    def randomize_catalog(silver_ruta: str, parametros: dict, **context) -> str:
        params = context["params"]

        destino = replicas.nulo_path(
            seed=params["seed"],
            n_repeticiones=params["n_randomizaciones"],
            **consulta(params),
        )
        fuente = Path(silver_ruta)

        # La semilla y las repeticiones ya están en el nombre del archivo, así
        # que alcanza con comparar contra silver, que es el único insumo.
        if (
            destino.exists()
            and not params["force"]
            and destino.stat().st_mtime >= fuente.stat().st_mtime
        ):
            log.info("Se reutilizó un catálogo nulo ya persistido.")
            return str(destino)

        nulo = replicas.randomize_catalog(
            silver_read(fuente),
            b=parametros["b"],
            d=parametros["d"],
            mc=parametros["mc"],
            n_repeticiones=params["n_randomizaciones"],
            seed=params["seed"],
        )
        replicas.nulo_write(destino, nulo)
        return str(destino)

    @task
    def thinning(vecinos_ruta: str, nulo_ruta: str, umbral_ruta: str, **context) -> str:
        params = context["params"]

        destino = replicas.replicas_path(seed=params["seed"], **consulta(params))
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

    @task
    def build_clusters(replicas_ruta: str, **context) -> dict[str, str]:
        params = context["params"]

        claves = {"seed": params["seed"], **consulta(params)}
        destino_eventos = clusters.eventos_path(**claves)
        destino_resumen = clusters.resumen_path(**claves)
        fuente = Path(replicas_ruta)

        salida = {"eventos": str(destino_eventos), "resumen": str(destino_resumen)}

        if (
            destino_eventos.exists()
            and destino_resumen.exists()
            and not params["force"]
            and min(destino_eventos.stat().st_mtime, destino_resumen.stat().st_mtime)
            >= fuente.stat().st_mtime
        ):
            log.info("Se reutilizaron los clusters ya armados.")
            return salida

        eventos, resumen = clusters.build_clusters(
            replicas.replicas_read(fuente),
            definicion_mainshock=params["mainshock"],
        )
        clusters.clusters_write(destino_eventos, eventos)
        clusters.clusters_write(destino_resumen, resumen)

        # No entra en el resultado: es el control que dice si los conteos son
        # creíbles, contrastados contra una ley que no salió de estos datos.
        log.info("Productividad: %s", clusters.productividad(resumen))
        return salida

    @task
    def publish_csv(clusters_rutas: dict[str, str], **context) -> dict[str, str]:
        """El entregable: los dos csv, y el chequeo de calidad sobre ellos.

        Es la única tarea que escribe csv. Las anteriores usan parquet porque se
        leen entre sí y ya vienen tipadas; ésta escribe el formato que se abre a
        mano para mirarlo y defenderlo.

        Y valida antes de dar la corrida por buena: si la clave repite, si
        quedaron menos filas de las que hacen falta o si alguna columna quedó
        entera en nulo, la tarea falla. Un dataset roto que se publica en verde
        es peor que una corrida en rojo, porque el error aparece recién cuando
        alguien ya construyó algo encima.
        """
        params = context["params"]
        claves = {"seed": params["seed"], **consulta(params)}

        destinos = {
            "dataset": str(entrega.dataset_path(**claves)),
            "eventos": str(entrega.eventos_path(**claves)),
        }

        informe = entrega.publicar(
            clusters.clusters_read(Path(clusters_rutas["eventos"])),
            clusters.clusters_read(Path(clusters_rutas["resumen"])),
            destinos,
        )

        log.info("Dataset publicado en %s | calidad: %s", destinos["dataset"], informe)
        return destinos

    silver_ruta = refine_silver(land_bronze())
    parametros = estimate_mc(silver_ruta)
    vecinos_ruta = nearest_neighbor(silver_ruta, parametros)

    publish_csv(
        build_clusters(
            thinning(
                vecinos_ruta,
                randomize_catalog(silver_ruta, parametros),
                fit_eta_threshold(vecinos_ruta),
            )
        )
    )


# Sin esta llamada el DAG no queda registrado: el decorador @dag sólo devuelve
# una factory, es invocarla lo que lo publica en el DagBag.
detector_replicas_api()
