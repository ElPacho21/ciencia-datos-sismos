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

import time

import pendulum
from airflow.providers.http.hooks.http import HttpHook

CONN_ID = "sismosapi"
EVENT_QUERY = "/fdsnws/event/1/query"
EVENT_COUNT = "/fdsnws/event/1/count"

# Tope duro del servicio. Se pide por debajo para dejar aire entre el conteo y
# la descarga: el catálogo puede crecer en el medio.
TOPE_SERVICIO = 20000
TOPE_PEDIDO = 18000

# Si una ventana de un minuto sigue sin entrar, partirla más no va a ayudar.
VENTANA_MINIMA_SEGUNDOS = 60


def _consulta(starttime, endtime, minmagnitude, eventtype, recorte, formato=None) -> dict:
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
    """Cuántos eventos empareja la consulta, sin bajarlos.

    El endpoint `count` devuelve un json de dos campos, así que preguntar sale
    mucho más barato que descargar y contar.
    """
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


def _partir(desde, hasta, eventtype, minmagnitude, recorte, timeout, retries) -> list:
    """Devuelve sub-ventanas [(desde, hasta), ...] que entren en el tope.

    Bisecta por tiempo en vez de repartir en partes iguales porque los sismos
    no se reparten parejo: una secuencia de réplicas puede meter miles de
    eventos en un par de días.
    """
    cantidad = contar(
        starttime=desde.to_iso8601_string(),
        endtime=hasta.to_iso8601_string(),
        eventtype=eventtype,
        minmagnitude=minmagnitude,
        recorte=recorte,
        timeout=timeout,
        retries=retries,
    )

    if cantidad == 0:
        return []

    if cantidad <= TOPE_PEDIDO:
        return [(desde, hasta)]

    if (hasta - desde).total_seconds() <= VENTANA_MINIMA_SEGUNDOS:
        raise ValueError(
            f"Entre {desde} y {hasta} hay {cantidad} eventos y la ventana ya no "
            f"se puede partir más. Subí `minmagnitude` para pedir menos."
        )

    medio = desde.add(seconds=(hasta - desde).total_seconds() / 2)
    return _partir(
        desde, medio, eventtype, minmagnitude, recorte, timeout, retries
    ) + _partir(medio, hasta, eventtype, minmagnitude, recorte, timeout, retries)


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
    """Baja el catálogo completo del rango pedido, partiéndolo si hace falta.

    El rectángulo (`minlatitude` y compañía) lo aplica el propio servicio. Sin
    él la consulta trae el mundo entero, que además de inútil hace que casi
    cualquier ventana larga choque contra el tope de 20000.

    `limit` no trunca: es un tope de seguridad. Si el rango empareja más
    eventos que eso, la tarea falla en vez de bajar un catálogo recortado a
    escondidas. `limit=0` significa sin tope.
    """
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
    ventanas = _partir(desde, hasta, eventtype, minmagnitude, recorte, timeout, retries)

    pedazos = []
    for indice, (ini, fin) in enumerate(ventanas):
        respuesta = _pedir(
            hook,
            EVENT_QUERY,
            _consulta(
                ini.to_iso8601_string(),
                fin.to_iso8601_string(),
                minmagnitude,
                eventtype,
                recorte,
                formato=format,
            ),
            timeout,
            retries,
        )
        contenido = respuesta.content

        # El encabezado del csv viene en cada pedazo: se conserva sólo el primero.
        if format == "csv" and indice > 0:
            _, _, cuerpo = contenido.partition(b"\n")
            contenido = cuerpo

        pedazos.append(contenido)

    return b"".join(pedazos)
