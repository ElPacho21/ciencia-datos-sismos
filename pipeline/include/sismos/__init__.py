"""Ingesta y refinado de sismos del servicio FDSN — Proyecto integrador de Ciencia de Datos."""

from pathlib import Path

OUTPUT_DIR = Path("/usr/local/airflow/include/output/detector_replicas_api")


def _numero(valor) -> str:
    """Normaliza un número para el nombre de archivo.

    Los `Param` de tipo number llegan como float, así que sin esto una corrida
    con magnitud 3 y otra con 3.0 caerían en archivos distintos y ninguna
    reutilizaría la caché de la otra.
    """
    numero = float(valor)
    return str(int(numero)) if numero.is_integer() else str(numero)


def particion(
    starttime,
    endtime,
    minmagnitude,
    minlatitude=None,
    maxlatitude=None,
    minlongitude=None,
    maxlongitude=None,
) -> str:
    """Nombre de partición al estilo Hive, compartido por todas las capas.

    Es la **única fuente de verdad de qué identifica a una corrida**: todo lo
    que cambie los datos de entrada tiene que aparecer acá. Si no, dos consultas
    distintas se pisan el mismo archivo y terminás analizando Argentina creyendo
    que es California.

    Las capas que además dependen de la semilla se la agregan por su cuenta.
    """
    nombre = "earthquake"

    if starttime is not None:
        nombre += f"_starttime={starttime}"

    if endtime is not None:
        nombre += f"_endtime={endtime}"

    if minmagnitude is not None:
        nombre += f"_minmagnitude={_numero(minmagnitude)}"

    # El rectángulo va en un solo campo y no en cuatro: cuatro pares
    # `clave=valor` más hacen nombres larguísimos, y esto ya cuelga de rutas
    # con prefijo de capa y sufijo de semilla.
    recorte = (minlatitude, maxlatitude, minlongitude, maxlongitude)
    if any(borde is not None for borde in recorte):
        nombre += "_bbox=" + ",".join(
            "" if borde is None else _numero(borde) for borde in recorte
        )

    return nombre
