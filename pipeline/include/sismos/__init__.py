"""Ingesta y refinado de sismos del servicio FDSN — Proyecto integrador de Ciencia de Datos."""

from pathlib import Path

OUTPUT_DIR = Path("/usr/local/airflow/include/output/detector_replicas_api")


def particion(starttime, endtime, minmagnitude) -> str:
    """Nombre de partición al estilo Hive, compartido por todas las capas.

    Bronze y silver de la misma consulta quedan con el mismo nombre (cambia
    sólo la extensión), así se emparejan de un vistazo y cada capa puede
    decidir por su cuenta si reutiliza lo que ya tiene.
    """
    nombre = "earthquake"

    if starttime is not None:
        nombre += f"_starttime={starttime}"

    if endtime is not None:
        nombre += f"_endtime={endtime}"

    if minmagnitude is not None:
        nombre += f"_minmagnitude={minmagnitude}"

    return nombre
