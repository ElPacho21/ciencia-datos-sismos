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
import time

import pendulum
from airflow.providers.http.hooks.http import HttpHook

log = logging.getLogger(__name__)

CONN_ID = "sismosapi"
EVENT_QUERY = "/fdsnws/event/1/query"
EVENT_COUNT = "/fdsnws/event/1/count"

# Lo que el servicio admite por consulta. Más que esto no lo trunca: lo rechaza
# con HTTP 400, así que hay que partir el pedido en varios.
TOPE_SERVICIO = 20000


def _consulta(
    starttime, endtime, minmagnitude, eventtype, recorte, formato=None
) -> dict:
    parametros = {
        "starttime": starttime,
        "endtime": endtime,
        "eventtype": eventtype,
        "minmagnitude": minmagnitude,
    }
    # El rectángulo lo aplica el servicio: filtrar después de bajar sería pedir
    # el mundo entero para tirar el 99%, y encima chocaría con el tope de 20000.
    parametros.update({k: v for k, v in (recorte or {}).items() if v is not None})

    if formato is not None:
        parametros["format"] = formato
    return parametros


def _pedir(hook, endpoint, parametros, timeout, retries):
    """Un pedido al servicio, con reintentos espaciados."""
    for intento in range(retries):
        try:
            respuesta = hook.run(
                endpoint=endpoint,
                data=parametros,
                extra_options={"check_response": True, "timeout": timeout},
            )
            return respuesta
        except Exception:
            if intento == retries - 1:
                raise
            time.sleep(2 * (intento + 1))


def contar(
    starttime,
    endtime,
    eventtype="earthquake",
    minmagnitude=0,
    recorte=None,
    timeout=60,
    retries=3,
) -> int:
    """Cuántos eventos empareja la consulta, sin bajarlos."""
    hook = HttpHook(method="GET", http_conn_id=CONN_ID)
    respuesta = _pedir(
        hook,
        EVENT_COUNT,
        _consulta(
            starttime, endtime, minmagnitude, eventtype, recorte, formato="geojson"
        ),
        timeout,
        retries,
    )
    return int(respuesta.json()["count"])


def _concatenar(pedazos: list[bytes]) -> bytes:
    """Pega varios csv en uno, dejando una sola cabecera.

    Cada respuesta del servicio trae la suya, así que las de los pedazos que no
    son el primero hay que sacarlas: si no, quedarían filas con la palabra
    'time' en el medio del archivo y silver las convertiría en nulos.
    """
    if len(pedazos) == 1:
        return pedazos[0]

    salida = [pedazos[0].rstrip(b"\n")]
    for pedazo in pedazos[1:]:
        cuerpo = pedazo.split(b"\n", 1)
        if len(cuerpo) == 2 and cuerpo[1].strip():
            salida.append(cuerpo[1].rstrip(b"\n"))

    return b"\n".join(salida) + b"\n"


def _bajar(
    hook, desde, hasta, total, consulta_base, recorte, timeout, retries
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
    if total <= TOPE_SERVICIO:
        respuesta = _pedir(
            hook,
            EVENT_QUERY,
            {**consulta_base, "starttime": desde, "endtime": hasta},
            timeout,
            retries,
        )
        return [respuesta.content]

    medio = desde + (hasta - desde) / 2

    # Si la ventana ya no se puede partir, el problema no tiene salida por acá:
    # hay más de 20000 eventos en un instante. Pasa sólo con rangos absurdos.
    if not desde < medio < hasta:
        raise ValueError(
            f"El rango {desde} a {hasta} empareja {total} sismos y ya no se "
            f"puede partir más. Subí `minmagnitude` o achicá la región."
        )

    total_izquierda = contar(
        starttime=desde,
        endtime=medio,
        eventtype=consulta_base["eventtype"],
        minmagnitude=consulta_base["minmagnitude"],
        recorte=recorte,
        timeout=timeout,
        retries=retries,
    )

    # Un evento justo en el borde puede caer en las dos mitades. No se corrige
    # acá: silver deduplica por `id`, y correr el borde un microsegundo abriría
    # la puerta a perder eventos, que es el error caro.
    return _bajar(
        hook, desde, medio, total_izquierda, consulta_base, recorte, timeout, retries
    ) + _bajar(
        hook,
        medio,
        hasta,
        total - total_izquierda,
        consulta_base,
        recorte,
        timeout,
        retries,
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
    timeout=60,
    retries=3,
) -> bytes:
    recorte = {
        "minlatitude": minlatitude,
        "maxlatitude": maxlatitude,
        "minlongitude": minlongitude,
        "maxlongitude": maxlongitude,
    }
    desde = pendulum.parse(str(starttime))
    hasta = pendulum.parse(str(endtime))

    if hasta <= desde:
        raise ValueError(f"La ventana está al revés o vacía: {starttime} a {endtime}.")

    total = contar(
        starttime=starttime,
        endtime=endtime,
        eventtype=eventtype,
        minmagnitude=minmagnitude,
        recorte=recorte,
        timeout=timeout,
        retries=retries,
    )

    if total == 0:
        raise ValueError(
            f"La consulta no empareja ningún sismo entre {starttime} y {endtime} "
            f"con magnitud mínima {minmagnitude}."
        )

    if limit and total > limit:
        raise ValueError(
            f"El rango empareja {total} sismos y el límite pedido es {limit}. "
            f"Subí `limit` para bajarlos todos, acortá la ventana o subí "
            f"`minmagnitude`. No se trunca en silencio a propósito: el catálogo "
            f"recortado daría un Mc y un b que no son los del período pedido."
        )

    hook = HttpHook(method="GET", http_conn_id=CONN_ID)

    # La consulta sin el rango de fechas: es lo único que cambia entre pedazos.
    consulta_base = _consulta(
        None, None, minmagnitude, eventtype, recorte, formato=format
    )
    consulta_base.pop("starttime")
    consulta_base.pop("endtime")

    pedazos = _bajar(
        hook, desde, hasta, total, consulta_base, recorte, timeout, retries
    )

    if len(pedazos) > 1:
        log.info(
            "El rango empareja %d sismos, más de los %d que admite el servicio "
            "por consulta: se bajó en %d pedidos.",
            total,
            TOPE_SERVICIO,
            len(pedazos),
        )

    return _concatenar(pedazos)
