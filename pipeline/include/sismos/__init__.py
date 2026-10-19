"""Ingesta y refinado de sismos del servicio FDSN — Proyecto integrador de Ciencia de Datos."""

from pathlib import Path

OUTPUT_DIR = Path("/usr/local/airflow/include/output/detector_replicas_api")


def _number(value) -> str:
    """Normaliza un número para el nombre de archivo.

    Los `Param` de tipo number llegan como float, así que sin esto una corrida
    con magnitud 3 y otra con 3.0 caerían en archivos distintos y ninguna
    reutilizaría la caché de la otra.
    """
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def method_suffix(r_max_km=None, r_max_mode="fixed", r_max_k=3.0) -> str:
    """Sufijo de nombre para los parámetros del *método* que cambian la salida.

    `partition()` identifica los datos de entrada y lo comparten bronze y
    silver; el tope de distancia padre-hijo recién actúa desde el vecino más
    cercano, así que no puede colarse ahí o se duplicaría el catálogo crudo.

    Con los valores por defecto (sin tope) devuelve `""`: los nombres de la
    corrida de referencia quedan idénticos a los de antes.
    """
    if r_max_mode not in ("fixed", "gk"):
        raise ValueError(f"Unknown r_max_mode: {r_max_mode!r}. Use 'fixed' or 'gk'.")

    if r_max_mode == "gk":
        suffix = f"_rmax=gk{_number(r_max_k)}"
        if r_max_km is not None:
            suffix += f"-cap{_number(r_max_km)}"
        return suffix

    if r_max_km is None:
        return ""
    return f"_rmax={_number(r_max_km)}"


def partition(
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
    name = "earthquake"

    if starttime is not None:
        name += f"_starttime={starttime}"

    if endtime is not None:
        name += f"_endtime={endtime}"

    if minmagnitude is not None:
        name += f"_minmagnitude={_number(minmagnitude)}"

    # El rectángulo va en un solo campo y no en cuatro: cuatro pares
    # `clave=valor` más hacen nombres larguísimos, y esto ya cuelga de rutas
    # con prefijo de capa y sufijo de semilla.
    bbox = (minlatitude, maxlatitude, minlongitude, maxlongitude)
    if any(edge is not None for edge in bbox):
        name += "_bbox=" + ",".join(
            "" if edge is None else _number(edge) for edge in bbox
        )

    return name
