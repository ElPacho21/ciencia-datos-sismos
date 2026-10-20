"""Corre el detector de réplicas en varias regiones y concatena las secuencias.

El método de Zaliapin & Ben-Zion supone un catálogo homogéneo, así que no se
corre sobre el mundo entero: cada región de `include/sismos/regions.yaml` se
procesa por separado, con su propio Mc y su propio eta_0, y recién después se
concatenan las secuencias en un solo csv con una columna `region`.

Este DAG no reimplementa nada del método: dispara `detector_replicas_api` una
vez por región (dynamic task mapping sobre el operador que dispara DAGs) y
espera a que terminen. Así se reutilizan sin tocar su caché, sus nombres de
archivo y sus validaciones; la bbox y las fechas ya distinguen las salidas de
cada región. Cuando todas terminaron, `combine_regions` lee los csv publicados,
valida que ninguna secuencia pertenezca a dos regiones y publica el combinado.
"""

from __future__ import annotations

import logging

import pandas as pd
import pendulum
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, task

from sismos import delivery
from sismos.features import PRIOR_RADIUS_KM, delivery_suffix
from sismos.regions import BORDER_BUFFER_KM, combine_regions, load_regions, regions_tag

log = logging.getLogger(__name__)

REGION_DAG_ID = "detector_replicas_api"

# Parámetros de método que se reenvían tal cual a cada corrida de región.
METHOD_PARAMS = (
    "mainshock",
    "seed",
    "n_randomizations",
    "r_max_km",
    "r_max_mode",
    "r_max_k",
    "prior_radius_km",
    "border_buffer_km",
    "force",
)


@dag(
    dag_id="detector_replicas_regions",
    schedule=None,
    start_date=pendulum.datetime(2026, 8, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["data-science", "capstone-project"],
    doc_md=__doc__,
    params={
        "mainshock": Param("largest", type="string", enum=["largest", "root"]),
        "seed": Param(1, type="integer", minimum=0),
        "n_randomizations": Param(5, type="integer", minimum=1, maximum=50),
        "r_max_km": Param(None, type=["number", "null"], minimum=1),
        "r_max_mode": Param("fixed", type="string", enum=["fixed", "gk"]),
        "r_max_k": Param(3.0, type="number", minimum=0.1),
        "prior_radius_km": Param(PRIOR_RADIUS_KM, type="number", minimum=1),
        "border_buffer_km": Param(BORDER_BUFFER_KM, type="number", minimum=0),
        "force": Param(False, type="boolean"),
    },
)
def detector_replicas_regions():
    @task
    def region_confs(**context) -> list[dict]:
        """Un `conf` por región: su rectángulo y ventana más los parámetros de método."""
        params = context["params"]
        shared = {k: params[k] for k in METHOD_PARAMS}

        confs = []
        for region in load_regions():
            confs.append(
                {
                    "starttime": str(region["starttime"]),
                    "endtime": str(region["endtime"]),
                    "minmagnitude": region["minmagnitude"],
                    "minlatitude": region["minlatitude"],
                    "maxlatitude": region["maxlatitude"],
                    "minlongitude": region["minlongitude"],
                    "maxlongitude": region["maxlongitude"],
                    # Una ventana larga supera los 20000 eventos: sin tope, el
                    # cliente la parte en varios pedidos en vez de fallar.
                    "limit": 0,
                    **shared,
                }
            )
        return confs

    run_regions = TriggerDagRunOperator.partial(
        task_id="run_region",
        trigger_dag_id=REGION_DAG_ID,
        wait_for_completion=True,
        poke_interval=30,
        failed_states=["failed"],
    ).expand(conf=region_confs())

    @task
    def combine(**context) -> str:
        """Concatena los csv publicados de cada región y publica el combinado."""
        params = context["params"]
        regions = load_regions()
        method = delivery_suffix(
            params["r_max_km"],
            params["r_max_mode"],
            params["r_max_k"],
            params["prior_radius_km"],
            params["border_buffer_km"],
        )

        frames = {}
        for region in regions:
            path = delivery.dataset_path(
                seed=params["seed"],
                mainshock=params["mainshock"],
                method=method,
                starttime=str(region["starttime"]),
                endtime=str(region["endtime"]),
                minmagnitude=region["minmagnitude"],
                minlatitude=region["minlatitude"],
                maxlatitude=region["maxlatitude"],
                minlongitude=region["minlongitude"],
                maxlongitude=region["maxlongitude"],
            )
            frames[region["name"]] = pd.read_csv(path)

        combined = combine_regions(frames)

        destination = delivery.combined_path(
            regions_tag(regions), params["seed"], params["mainshock"], method
        )
        delivery.write_csv(destination, combined)

        report, problems = delivery.check(combined, "cluster_id", "combined")
        if problems:
            raise ValueError(
                "The combined dataset does not meet the quality criteria:\n  - "
                + "\n  - ".join(problems)
            )

        log.info("Combined dataset published to %s | quality: %s", destination, report)
        return str(destination)

    run_regions >> combine()


detector_replicas_regions()
