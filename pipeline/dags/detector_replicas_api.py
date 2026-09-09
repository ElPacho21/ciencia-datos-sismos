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
`n_aftershocks`. Se publica además el catálogo evento por evento, donde una fila
es un terremoto y el objetivo es `is_aftershock`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pendulum
from airflow.sdk import Param, dag, task

# Se importan los módulos enteros porque varias tareas del DAG se llaman igual
# que la función que las hace, y de otro modo una taparía a la otra.
from sismos import aftershocks, clusters, delivery, neighbors
from sismos.bronze import bronze_load, bronze_path, bronze_write
from sismos.parameters import estimate
from sismos.silver import refine, silver_path, silver_read, silver_write
from sismos.threshold import fit_threshold, threshold_path, threshold_read, threshold_write
from sismos.usgs_earthquake import fetch

log = logging.getLogger(__name__)


@dag(
    dag_id="detector_replicas_api",
    schedule=None,
    start_date=pendulum.datetime(2026, 8, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["data-science", "capstone-project"],
    doc_md=__doc__,
    params={
        "starttime": Param(
            "2016-01-01",
            type="string",
            format="date",
            title="Start date",
            description=("Date from which to fetch earthquake records."),
        ),
        "endtime": Param(
            "2026-01-01",
            type="string",
            format="date",
            title="End date",
            description=("Date up to which to fetch earthquake records."),
        ),
        "minmagnitude": Param(
            3.5,
            type="number",
            title="Minimum magnitude",
            description="Minimum magnitude of the queried earthquakes.",
        ),
        # Rectángulo por defecto: Argentina continental. El método supone un
        # catálogo homogéneo, y el mundo entero no lo es — mezcla regiones con
        # completitudes muy distintas y hace que los árboles de réplicas
        # encadenen zonas sin relación. Dejar los cuatro en null vuelve a
        # consultar el planeta completo.
        "minlatitude": Param(
            -55,
            type=["number", "null"],
            title="Minimum latitude",
            description="Southern edge of the queried rectangle.",
            minimum=-90,
            maximum=90,
        ),
        "maxlatitude": Param(
            -21,
            type=["number", "null"],
            title="Maximum latitude",
            description="Northern edge of the queried rectangle.",
            minimum=-90,
            maximum=90,
        ),
        "minlongitude": Param(
            -74,
            type=["number", "null"],
            title="Minimum longitude",
            description="Western edge of the queried rectangle.",
            minimum=-180,
            maximum=180,
        ),
        "maxlongitude": Param(
            -53,
            type=["number", "null"],
            title="Maximum longitude",
            description="Eastern edge of the queried rectangle.",
            minimum=-180,
            maximum=180,
        ),
        "limit": Param(
            20000,
            type="integer",
            title="Cap on earthquakes to download",
            description=(
                "A safety cap, not a filter: if the range matches more "
                "earthquakes than this, the run fails instead of downloading a "
                "truncated catalog. 0 means no cap — the whole window is "
                "downloaded, split into several requests if it exceeds the "
                "20000 the service allows per query."
            ),
            minimum=0,
            maximum=20000,
        ),
        "mainshock": Param(
            "largest",
            type="string",
            enum=["largest", "root"],
            title="Mainshock definition",
            description=(
                "Which of the cluster's events is the mainshock. 'largest' is "
                "the one with the highest magnitude, which is what Zaliapin & "
                "Ben-Zion use; 'root' is the one that triggered the sequence. "
                "They differ when there were foreshocks."
            ),
        ),
        "seed": Param(
            1,
            type="integer",
            title="Seed",
            description=(
                "Thinning is probabilistic: it shuffles the catalog and draws "
                "which links go back to the background. Without fixing the seed "
                "the results are not reproducible, so it is recorded in the "
                "name of the files it produces."
            ),
            minimum=0,
        ),
        "n_randomizations": Param(
            5,
            type="integer",
            title="Null-catalog shuffles",
            description=(
                "How many times the catalog is shuffled to estimate the "
                "distribution of eta under pure chance. More shuffles give a "
                "less noisy null, but each one is a full run of the nearest "
                "neighbor."
            ),
            minimum=1,
            maximum=50,
        ),
        "force": Param(
            False,
            type="boolean",
            title="Force the run",
            description=(
                "Ignores every cache: downloads even if the source has not "
                "changed, and re-requests the csv already in bronze."
            ),
        ),
    },
)
def detector_replicas_api():
    def query(params) -> dict:
        return {
            k: params[k]
            for k in (
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

        destination = bronze_path(**query(params))

        if destination.exists() and not params["force"]:
            log.info("Reused a csv already persisted.")
            return str(destination)

        csv = fetch(limit=params["limit"], **query(params))
        bronze_write(destination, csv)
        log.info("Downloaded the csv from the API to %s.", destination)
        return str(destination)

    @task
    def refine_silver(bronze_file: str, **context) -> str:
        params = context["params"]

        destination = silver_path(**query(params))

        if destination.exists() and not params["force"]:
            log.info("Reused a silver parquet already persisted.")
            return str(destination)

        events = refine(bronze_load(Path(bronze_file)))
        silver_write(destination, events)
        log.info("Refined %d events to %s.", len(events), destination)
        return str(destination)

    @task
    def estimate_mc(silver_file: str) -> dict:
        """Mc del catálogo, más los b y d estándar."""
        return estimate(silver_read(Path(silver_file)))

    @task
    def nearest_neighbor(silver_file: str, parameters: dict, **context) -> str:
        params = context["params"]

        destination = neighbors.neighbors_path(**query(params))
        source = Path(silver_file)

        if (
            destination.exists()
            and not params["force"]
            and destination.stat().st_mtime >= source.stat().st_mtime
        ):
            log.info("Reused a parent forest already persisted.")
            return str(destination)

        paired = neighbors.nearest_neighbor(
            silver_read(source),
            b=parameters["b"],
            d=parameters["d"],
            mc=parameters["mc"],
        )
        neighbors.neighbors_write(destination, paired)
        return str(destination)

    @task
    def fit_eta_threshold(neighbors_file: str, **context) -> str:
        params = context["params"]

        destination = threshold_path(**query(params))
        source = Path(neighbors_file)

        if (
            destination.exists()
            and not params["force"]
            and destination.stat().st_mtime >= source.stat().st_mtime
        ):
            log.info("Reused a threshold already fitted.")
            return str(destination)

        threshold = fit_threshold(neighbors.neighbors_read(source))
        threshold_write(destination, threshold)
        return str(destination)

    @task
    def randomize_catalog(silver_file: str, parameters: dict, **context) -> str:
        params = context["params"]

        destination = aftershocks.null_path(
            seed=params["seed"],
            n_repetitions=params["n_randomizations"],
            **query(params),
        )
        source = Path(silver_file)

        # La semilla y las repeticiones ya están en el nombre del archivo, así
        # que alcanza con comparar contra silver, que es el único insumo.
        if (
            destination.exists()
            and not params["force"]
            and destination.stat().st_mtime >= source.stat().st_mtime
        ):
            log.info("Reused a null catalog already persisted.")
            return str(destination)

        null = aftershocks.randomize_catalog(
            silver_read(source),
            b=parameters["b"],
            d=parameters["d"],
            mc=parameters["mc"],
            n_repetitions=params["n_randomizations"],
            seed=params["seed"],
        )
        aftershocks.null_write(destination, null)
        return str(destination)

    @task
    def thinning(neighbors_file: str, null_file: str, threshold_file: str, **context) -> str:
        params = context["params"]

        destination = aftershocks.aftershocks_path(seed=params["seed"], **query(params))
        sources = [Path(neighbors_file), Path(null_file), Path(threshold_file)]

        if (
            destination.exists()
            and not params["force"]
            and all(destination.stat().st_mtime >= f.stat().st_mtime for f in sources)
        ):
            log.info("Reused an aftershock classification already persisted.")
            return str(destination)

        threshold = threshold_read(Path(threshold_file))
        paired = neighbors.neighbors_read(Path(neighbors_file))

        # El nulo no entra en la clasificación: sirve para saber si la
        # bimodalidad que se está usando para clasificar es real.
        contrast = aftershocks.contrast_with_null(
            paired,
            aftershocks.null_read(Path(null_file)),
            log10_eta0=threshold["log10_eta0"],
        )

        # El sorteo usa una semilla derivada de la del DAG: barajar el catálogo
        # y sortear los enlaces son cosas distintas y no comparten el flujo.
        classified, diagnostics = aftershocks.thin(
            paired,
            mixture=threshold["mixture"],
            log10_eta0=threshold["log10_eta0"],
            seed=params["seed"] + 1,
        )
        aftershocks.aftershocks_write(destination, classified)
        log.info("Thinning: %s | contrast with the null: %s", diagnostics, contrast)
        return str(destination)

    @task
    def build_clusters(aftershocks_file: str, **context) -> dict[str, str]:
        params = context["params"]

        keys = {
            "seed": params["seed"],
            "mainshock": params["mainshock"],
            **query(params),
        }
        destination_events = clusters.events_path(**keys)
        destination_summary = clusters.summary_path(**keys)
        source = Path(aftershocks_file)

        output = {"events": str(destination_events), "summary": str(destination_summary)}

        if (
            destination_events.exists()
            and destination_summary.exists()
            and not params["force"]
            and min(destination_events.stat().st_mtime, destination_summary.stat().st_mtime)
            >= source.stat().st_mtime
        ):
            log.info("Reused the clusters already built.")
            return output

        events, summary = clusters.build_clusters(
            aftershocks.aftershocks_read(source),
            mainshock_definition=params["mainshock"],
        )
        clusters.clusters_write(destination_events, events)
        clusters.clusters_write(destination_summary, summary)

        # No entra en el resultado: es el control que dice si los conteos son
        # creíbles, contrastados contra una ley que no salió de estos datos.
        log.info("Productivity: %s", clusters.productivity(summary))
        return output

    @task
    def publish_csv(clusters_files: dict[str, str], **context) -> dict[str, str]:
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
        keys = {
            "seed": params["seed"],
            "mainshock": params["mainshock"],
            **query(params),
        }

        destinations = {
            "dataset": str(delivery.dataset_path(**keys)),
            "events": str(delivery.events_path(**keys)),
        }

        report = delivery.publish(
            clusters.clusters_read(Path(clusters_files["events"])),
            clusters.clusters_read(Path(clusters_files["summary"])),
            destinations,
        )

        log.info("Dataset published to %s | quality: %s", destinations["dataset"], report)
        return destinations

    silver_file = refine_silver(land_bronze())
    parameters = estimate_mc(silver_file)
    neighbors_file = nearest_neighbor(silver_file, parameters)

    publish_csv(
        build_clusters(
            thinning(
                neighbors_file,
                randomize_catalog(silver_file, parameters),
                fit_eta_threshold(neighbors_file),
            )
        )
    )


# Sin esta llamada el DAG no queda registrado: el decorador @dag sólo devuelve
# una factory, es invocarla lo que lo publica en el DagBag.
detector_replicas_api()
