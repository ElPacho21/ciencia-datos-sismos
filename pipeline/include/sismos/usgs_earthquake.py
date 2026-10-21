"""Cliente del servicio FDSN de eventos del USGS.

El servicio corta cualquier consulta que empareje más de 20000 eventos, y no
trunca: devuelve HTTP 400. Pasarle `limit` evita el error, pero es peor el
remedio — el FDSN ordena por tiempo descendente, así que devolvería los 20000
más recientes y el CSV cubriría una ventana más corta que la pedida, sin decirlo.
El nombre del archivo en bronze diría un `starttime` que no es cierto y todo el
análisis saldría sobre un catálogo recortado en silencio.

Así que acá se hace al revés: se pregunta primero cuántos eventos hay y, si no
entran, se parte el rango de fechas por la mitad hasta que cada pedido entre.
Los pedazos se concatenan. Un evento justo en el borde entre dos sub-ventanas
puede venir repetido, pero silver deduplica por `id`, así que no hace daño.
"""

import logging
import random
import time

import pendulum
import requests
from airflow.providers.http.hooks.http import HttpHook

log = logging.getLogger(__name__)

CONN_ID = "usgs_fdsn"
EVENT_QUERY = "/fdsnws/event/1/query"
EVENT_COUNT = "/fdsnws/event/1/count"

# Lo que el servicio admite por consulta. Más que esto no lo trunca: lo rechaza
# con HTTP 400, así que hay que partir el pedido en varios.
SERVICE_LIMIT = 20000

# Cuántos pedidos se hacen como máximo ante un fallo transitorio, y cuánto se
# espera entre uno y otro. La espera se duplica por intento y se sortea dentro
# de una banda: cinco corridas disparadas a la vez no vuelven todas en el mismo
# instante a golpear el servicio. Techo para no quedarse colgado media hora.
ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 30.0

# Lo que el servicio tarda en armar una respuesta de 20000 eventos no es lo que
# tarda en contestar un conteo: con 60 s, un trozo grande del catálogo de Japón
# se cortaba por lectura y el reintento volvía a pedir lo mismo, con el mismo
# límite. El timeout es del pedido, no del servicio: si contesta antes, se
# sigue igual.
TIMEOUT_SECONDS = 180


def _query(
    starttime, endtime, minmagnitude, eventtype, bbox, fmt=None
) -> dict:
    params = {
        "starttime": starttime,
        "endtime": endtime,
        "eventtype": eventtype,
        "minmagnitude": minmagnitude,
    }
    # El rectángulo lo aplica el servicio: filtrar después de bajar sería pedir
    # el mundo entero para tirar el 99%, y encima chocaría con el tope de 20000.
    params.update({k: v for k, v in (bbox or {}).items() if v is not None})

    if fmt is not None:
        params["format"] = fmt
    return params


def _is_transient(error: Exception) -> bool:
    """Si el fallo es del servicio y vale la pena volver a pedir.

    Un 5xx o un 429 son transitorios: el mismo pedido puede salir bien un
    segundo después. Un 4xx dice que la consulta está mal armada —repetirla
    devuelve exactamente el mismo error— y una excepción que no viene de HTTP
    es un bug de acá: en los dos casos conviene fallar rápido en vez de
    demorar el diagnóstico.
    """
    if isinstance(
        error, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    ):
        return True
    # `HttpHook.check_response` convierte el status en AirflowException("502:Bad Gateway").
    code = str(error).partition(":")[0].strip()
    return code.isdigit() and (int(code) >= 500 or int(code) == 429)


def _backoff(attempt: int) -> float:
    """Espera antes del intento siguiente: 2, 4, 8, 16 s, con jitter y tope."""
    delay = min(BACKOFF_BASE_SECONDS * 2**attempt, BACKOFF_MAX_SECONDS)
    return random.uniform(delay / 2, delay)


def _request(hook, endpoint, params, timeout, attempts=ATTEMPTS):
    """Un pedido al servicio, con reintentos espaciados ante fallos transitorios."""
    for attempt in range(attempts):
        try:
            response = hook.run(
                endpoint=endpoint,
                data=params,
                extra_options={"check_response": True, "timeout": timeout},
            )
            return response
        except Exception as error:
            if attempt == attempts - 1 or not _is_transient(error):
                raise
            delay = _backoff(attempt)
            log.warning(
                "Attempt %d/%d to %s failed (%s): retrying in %.1f s.",
                attempt + 1,
                attempts,
                endpoint,
                error,
                delay,
            )
            time.sleep(delay)


def count(
    starttime,
    endtime,
    eventtype="earthquake",
    minmagnitude=0,
    bbox=None,
    timeout=TIMEOUT_SECONDS,
    attempts=ATTEMPTS,
) -> int:
    """Cuántos eventos empareja la consulta, sin bajarlos."""
    hook = HttpHook(method="GET", http_conn_id=CONN_ID)
    response = _request(
        hook,
        EVENT_COUNT,
        _query(
            starttime, endtime, minmagnitude, eventtype, bbox, fmt="geojson"
        ),
        timeout,
        attempts,
    )
    return int(response.json()["count"])


def _concatenate(chunks: list[bytes]) -> bytes:
    """Pega varios csv en uno, dejando una sola cabecera.

    Cada respuesta del servicio trae la suya, así que las de los pedazos que no
    son el primero hay que sacarlas: si no, quedarían filas con la palabra
    'time' en el medio del archivo y silver las convertiría en nulos.
    """
    if len(chunks) == 1:
        return chunks[0]

    out = [chunks[0].rstrip(b"\n")]
    for chunk in chunks[1:]:
        body = chunk.split(b"\n", 1)
        if len(body) == 2 and body[1].strip():
            out.append(body[1].rstrip(b"\n"))

    return b"\n".join(out) + b"\n"


def _download(
    hook, start, end, total, base_query, bbox, timeout, attempts
) -> list[bytes]:
    """Baja el rango, partiéndolo por la mitad mientras no entre en una consulta.

    Se bisecta en vez de repartir en partes iguales porque los sismos no se
    distribuyen parejo en el tiempo: una secuencia de réplicas mete miles de
    eventos en un par de días, y un reparto uniforme dejaría ese pedazo igual de
    grande que el original. Cortando por la mitad y volviendo a contar, la
    partición se adapta a dónde está la densidad.

    `total` viene contado por quien llama, así que cada nivel de la recursión
    cuesta **una** consulta al endpoint de conteo, no dos.
    """
    if total <= SERVICE_LIMIT:
        response = _request(
            hook,
            EVENT_QUERY,
            {**base_query, "starttime": start, "endtime": end},
            timeout,
            attempts,
        )
        return [response.content]

    mid = start + (end - start) / 2

    # Si la ventana ya no se puede partir, el problema no tiene salida por acá:
    # hay más de 20000 eventos en un instante. Pasa sólo con rangos absurdos.
    if not start < mid < end:
        raise ValueError(
            f"The range {start} to {end} matches {total} quakes and can no "
            f"longer be split. Raise `minmagnitude` or shrink the region."
        )

    left_total = count(
        starttime=start,
        endtime=mid,
        eventtype=base_query["eventtype"],
        minmagnitude=base_query["minmagnitude"],
        bbox=bbox,
        timeout=timeout,
        attempts=attempts,
    )

    # Un evento justo en el borde puede caer en las dos mitades. No se corrige
    # acá: silver deduplica por `id`, y correr el borde un microsegundo abriría
    # la puerta a perder eventos, que es el error caro.
    return _download(
        hook, start, mid, left_total, base_query, bbox, timeout, attempts
    ) + _download(
        hook,
        mid,
        end,
        total - left_total,
        base_query,
        bbox,
        timeout,
        attempts,
    )


def fetch(
    starttime,
    endtime,
    format="csv",
    eventtype="earthquake",
    minmagnitude=0,
    limit=0,
    minlatitude=None,
    maxlatitude=None,
    minlongitude=None,
    maxlongitude=None,
    timeout=TIMEOUT_SECONDS,
    attempts=ATTEMPTS,
) -> bytes:
    bbox = {
        "minlatitude": minlatitude,
        "maxlatitude": maxlatitude,
        "minlongitude": minlongitude,
        "maxlongitude": maxlongitude,
    }
    start = pendulum.parse(str(starttime))
    end = pendulum.parse(str(endtime))

    if end <= start:
        raise ValueError(f"The window is reversed or empty: {starttime} to {endtime}.")

    total = count(
        starttime=starttime,
        endtime=endtime,
        eventtype=eventtype,
        minmagnitude=minmagnitude,
        bbox=bbox,
        timeout=timeout,
        attempts=attempts,
    )

    if total == 0:
        raise ValueError(
            f"The query matches no quakes between {starttime} and {endtime} "
            f"with minimum magnitude {minmagnitude}."
        )

    if limit and total > limit:
        raise ValueError(
            f"The range matches {total} quakes and the requested limit is {limit}. "
            f"Raise `limit` to download them all, shorten the window or raise "
            f"`minmagnitude`. It is not truncated silently on purpose: the "
            f"clipped catalog would give an Mc and a b that are not those of "
            f"the requested period."
        )

    hook = HttpHook(method="GET", http_conn_id=CONN_ID)

    # La consulta sin el rango de fechas: es lo único que cambia entre pedazos.
    base_query = _query(
        None, None, minmagnitude, eventtype, bbox, fmt=format
    )
    base_query.pop("starttime")
    base_query.pop("endtime")

    chunks = _download(
        hook, start, end, total, base_query, bbox, timeout, attempts
    )

    if len(chunks) > 1:
        log.info(
            "The range matches %d quakes, more than the %d the service allows "
            "per query: downloaded in %d requests.",
            total,
            SERVICE_LIMIT,
            len(chunks),
        )

    return _concatenate(chunks)
