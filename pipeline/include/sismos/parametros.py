"""Estimación de los parámetros del método de Zaliapin & Ben-Zion.

Tres números que salen del propio catálogo y que la distancia de vecino más
cercano necesita antes de poder calcularse:

    eta_ij = t_ij * (r_ij ** d) * 10 ** (-b * m_i)

- `mc`: magnitud de completitud. Por debajo de Mc el catálogo pierde eventos,
  así que tanto b como d se estiman recién después de recortar ahí.
- `b`: pendiente de Gutenberg-Richter por máxima verosimilitud (Aki 1965), con
  la corrección de binning de Utsu y el error de Shi & Bolt (1982).
- `d`: dimensión de correlación de los epicentros (Grassberger-Procaccia), que
  es lo que Zaliapin & Ben-Zion usan como dimensión fractal.

No son constantes: son parámetros ajustados a *estos* datos. Por eso el JSON
que sale de acá guarda también las curvas diagnósticas (la FMD y la integral
de correlación), para poder graficarlas y defender los valores sin recalcular.

Advertencia de método: sobre un catálogo global el Mc es muy heterogéneo entre
regiones y la integral de correlación queda dominada por la geometría de los
bordes de placa. Estos números son sólidos sobre una región acotada; sobre el
mundo entero hay que leerlos con pinzas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, particion
from sismos.geo import CHUNK_DISTANCIAS, distancias_epicentrales

log = logging.getLogger(__name__)

PARAMETROS_DIR = OUTPUT_DIR / "parametros"

# El USGS publica las magnitudes en una grilla de 0.1 (con excepciones que se
# redondean), y el binning entra en la corrección de Aki.
BIN_MAGNITUD = 0.1

LOG10_E = float(np.log10(np.e))


def recortar_a_mc(
    sismos: pd.DataFrame, mc: float, bin_magnitud: float = BIN_MAGNITUD
) -> pd.DataFrame:
    """Se queda con los eventos completos, los de magnitud >= Mc.

    El corte va sobre las magnitudes binneadas, igual que adentro de `b_aki`:
    con las crudas queda corrido medio bin y no todos los pasos terminarían
    trabajando exactamente sobre el mismo subcatálogo.
    """
    binneadas = np.round(sismos["mag"].to_numpy(dtype=float) / bin_magnitud)
    return sismos[binneadas * bin_magnitud >= mc - bin_magnitud / 4]


def parametros_path(**consulta) -> Path:
    """Misma partición que bronze y silver, pero en json.

    Es un puñado de números y dos curvas: no justifica parquet, y en json se
    puede abrir a mano para pegarlo en el informe.
    """
    return PARAMETROS_DIR / f"{particion(**consulta)}.json"


def parametros_write(destino: Path, parametros: dict) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(parametros, indent=2), encoding="utf-8")


def parametros_read(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def fmd(magnitudes: np.ndarray, bin_magnitud: float = BIN_MAGNITUD):
    """Distribución de frecuencia-magnitud, no acumulada y acumulada.

    Devuelve los centros de bin, cuántos eventos caen en cada uno y cuántos hay
    de esa magnitud para arriba (que es la forma en que se escribe
    Gutenberg-Richter).
    """
    m = np.round(magnitudes / bin_magnitud)
    indices = (m - m.min()).astype(int)
    no_acumulada = np.bincount(indices)
    centros = np.round((m.min() + np.arange(len(no_acumulada))) * bin_magnitud, 4)

    # Acumulada "de acá para arriba": se suma desde el final hacia atrás.
    acumulada = np.cumsum(no_acumulada[::-1])[::-1]

    return centros, no_acumulada, acumulada


def b_aki(magnitudes: np.ndarray, mc: float, bin_magnitud: float = BIN_MAGNITUD):
    """b de Gutenberg-Richter por máxima verosimilitud sobre los eventos >= Mc.

    El `- bin/2` es la corrección de Utsu: como las magnitudes vienen
    redondeadas al bin, el mínimo real de la muestra está medio bin por debajo
    del Mc nominal. Sin eso, b sale sistemáticamente alto.
    """
    m = np.round(magnitudes / bin_magnitud) * bin_magnitud
    m = m[m >= mc - bin_magnitud / 4]
    n = len(m)

    if n < 2:
        raise ValueError(f"Sólo {n} eventos por encima de Mc={mc}: no alcanza para b.")

    media = float(m.mean())
    denominador = media - (mc - bin_magnitud / 2)
    if denominador <= 0:
        raise ValueError(
            f"La magnitud media ({media:.3f}) no supera a Mc={mc}: el catálogo "
            "no sigue Gutenberg-Richter en ese corte."
        )

    b = LOG10_E / denominador

    # Shi & Bolt (1982): el error de b no es 1/sqrt(n), depende de la dispersión
    # de las magnitudes.
    sigma = 2.30 * b**2 * float(np.sqrt(((m - media) ** 2).sum() / (n * (n - 1))))

    return b, sigma, n


def mc_maxc(
    centros: np.ndarray,
    no_acumulada: np.ndarray,
    correccion: float = 0.2,
) -> float:
    """Máxima curvatura: el bin más poblado de la FMD no acumulada.

    Es donde el catálogo deja de crecer y empieza a perder eventos. Sabe
    subestimar, así que se le suma la corrección empírica de Woessner & Wiemer
    (2005).
    """
    return float(centros[int(np.argmax(no_acumulada))] + correccion)


def mc_gft(
    centros: np.ndarray,
    acumulada: np.ndarray,
    magnitudes: np.ndarray,
    bin_magnitud: float = BIN_MAGNITUD,
    objetivo: float = 90.0,
    min_eventos: int = 50,
):
    """Bondad de ajuste (Wiemer & Wyss 2000).

    Para cada Mc candidato se ajusta Gutenberg-Richter y se compara la FMD
    sintética contra la observada. Se queda con el Mc *más chico* que explique
    al menos `objetivo`% de los datos: cuanto más bajo el corte, más eventos
    sobreviven para el resto del análisis.

    Devuelve `(None, mejor_R)` si ningún candidato llega al objetivo, para que
    quien llame decida el fallback.
    """
    mejor_r = -np.inf

    for i, candidato in enumerate(centros):
        if acumulada[i] < min_eventos:
            break

        try:
            b, _, n = b_aki(magnitudes, float(candidato), bin_magnitud)
        except ValueError:
            continue

        a = np.log10(n) + b * candidato

        observada = acumulada[i:]
        sintetica = 10 ** (a - b * centros[i:])
        r = 100.0 - 100.0 * float(
            np.abs(observada - sintetica).sum() / observada.sum()
        )

        mejor_r = max(mejor_r, r)

        if r >= objetivo:
            return float(candidato), r

    return None, (float(mejor_r) if np.isfinite(mejor_r) else None)


def correlation_curve(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    radios: np.ndarray,
    chunk: int = CHUNK_DISTANCIAS,
) -> np.ndarray:
    """Integral de correlación C(r): fracción de pares más cercanos que r.

    Va por bloques y acumula un histograma en vez de guardar las distancias:
    un catálogo de 20000 eventos son 4e8 pares, que como float64 no entran en
    memoria.
    """
    n = len(latitudes)
    if n < 2:
        raise ValueError("Hacen falta al menos dos eventos para la integral.")

    lat_rad = np.radians(latitudes.astype(float))
    lon_rad = np.radians(longitudes.astype(float))

    bordes = np.concatenate([[0.0], radios])
    conteo = np.zeros(len(radios))

    for desde in range(0, n, chunk):
        hasta = min(desde + chunk, n)
        distancias = distancias_epicentrales(lat_rad, lon_rad, desde, hasta)
        histograma, _ = np.histogram(distancias, bins=bordes)
        conteo += np.cumsum(histograma)

        # La diagonal son las distancias de cada evento consigo mismo: valen 0
        # y entrarían en todos los radios, así que se descuentan.
        conteo -= hasta - desde

    # Se recorren los pares en ambos sentidos, de ahí el n*(n-1) en vez del
    # n*(n-1)/2 de la definición con pares no ordenados.
    return conteo / (n * (n - 1))


def d_correlation(
    radios: np.ndarray,
    curva: np.ndarray,
    min_puntos: int = 10,
    r2_minimo: float = 0.99,
):
    """Pendiente de log C(r) vs log r en su tramo recto.

    El rango de escaleo es la decisión delicada del método: abajo lo domina el
    error de localización y arriba la saturación (C -> 1). Se elige la ventana
    *más ancha* que se ajuste a una recta con R² >= `r2_minimo`, y no la de
    mejor R² a secas: como la ventana más angosta siempre es la más recta,
    maximizar R² devuelve un tramo de una década pegado al extremo bajo, que
    describe el apretujamiento de las réplicas y no la geometría del catálogo.
    Si ninguna ventana llega al piso se cae a la de mejor R², avisando.

    Devuelve el rango elegido para poder auditarlo contra el gráfico.
    """
    usables = (curva > 0) & (curva < 1)
    if usables.sum() < min_puntos:
        raise ValueError(
            "La integral de correlación no tiene suficientes puntos utilizables; "
            "probá con un catálogo más grande o más radios."
        )

    x = np.log10(radios[usables])
    y = np.log10(curva[usables])

    mas_ancha = None
    mejor_r2 = None

    for i in range(len(x) - min_puntos + 1):
        for j in range(i + min_puntos, len(x) + 1):
            pendiente, ordenada = np.polyfit(x[i:j], y[i:j], 1)
            residuos = y[i:j] - (pendiente * x[i:j] + ordenada)
            varianza = ((y[i:j] - y[i:j].mean()) ** 2).sum()
            r2 = 1.0 - (residuos**2).sum() / varianza if varianza > 0 else 0.0

            candidato = (float(pendiente), float(r2), (i, j))

            if mejor_r2 is None or r2 > mejor_r2[1]:
                mejor_r2 = candidato

            # Entre las que pasan el piso gana la más ancha; a igual ancho, la
            # de mejor ajuste.
            if r2 >= r2_minimo and (
                mas_ancha is None
                or (j - i, r2) > (mas_ancha[2][1] - mas_ancha[2][0], mas_ancha[1])
            ):
                mas_ancha = candidato

    if mas_ancha is None:
        assert mejor_r2 is not None
        log.warning(
            "Ningún tramo de la integral de correlación llegó a R²=%.3f; se usa "
            "el de mejor ajuste (R²=%.4f). Conviene mirar la curva a mano.",
            r2_minimo,
            mejor_r2[1],
        )
        elegido = mejor_r2
    else:
        elegido = mas_ancha

    d, r2, (i, j) = elegido
    rango = (float(radios[usables][i]), float(radios[usables][j - 1]))

    return d, r2, rango


def estimate(
    sismos: pd.DataFrame,
    mc_metodo: str = "gft",
    bin_magnitud: float = BIN_MAGNITUD,
    correccion_maxc: float = 0.2,
    gft_objetivo: float = 90.0,
    min_eventos: int = 50,
    n_radios: int = 50,
    min_puntos_ajuste: int = 10,
    d_r2_minimo: float = 0.99,
) -> dict:
    """Estima Mc, b y d sobre el catálogo silver y devuelve el reporte completo."""
    if mc_metodo not in ("gft", "maxc"):
        raise ValueError(f"Método de Mc desconocido: {mc_metodo!r}. Usá 'gft' o 'maxc'.")

    magnitudes = sismos["mag"].to_numpy(dtype=float)
    centros, no_acumulada, acumulada = fmd(magnitudes, bin_magnitud)

    maxc = mc_maxc(centros, no_acumulada, correccion_maxc)
    gft, gft_r = mc_gft(
        centros, acumulada, magnitudes, bin_magnitud, gft_objetivo, min_eventos
    )

    if mc_metodo == "maxc":
        mc = maxc
    elif gft is not None:
        mc = gft
    else:
        mc = maxc
        log.warning(
            "Ningún Mc llegó al %.0f%% de bondad de ajuste (mejor: %s). "
            "Se cae a máxima curvatura, Mc=%.2f.",
            gft_objetivo,
            f"{gft_r:.1f}%" if gft_r is not None else "n/d",
            mc,
        )

    completos = recortar_a_mc(sismos, mc, bin_magnitud)
    if len(completos) < min_eventos:
        raise ValueError(
            f"Sólo quedan {len(completos)} eventos por encima de Mc={mc:.2f} "
            f"(mínimo {min_eventos}). Ampliá la ventana o bajá minmagnitude."
        )

    b, b_sigma, n_b = b_aki(completos["mag"].to_numpy(dtype=float), mc, bin_magnitud)
    a = float(np.log10(n_b) + b * mc)

    # b fuera de este rango no es un temblor raro, es un síntoma: casi siempre
    # significa que Mc quedó por debajo de la completitud real y la FMD se
    # aplanó porque al catálogo le faltan los eventos chicos.
    if not 0.5 <= b <= 2.0:
        log.warning(
            "b=%.3f está fuera del rango físico habitual (0.5-2.0) con Mc=%.2f. "
            "Revisá la FMD: lo más probable es que el Mc esté subestimado.",
            b,
            mc,
        )

    radios = np.logspace(np.log10(0.1), np.log10(20100.0), n_radios)
    curva = correlation_curve(
        completos["latitude"].to_numpy(), completos["longitude"].to_numpy(), radios
    )
    d, d_r2, d_rango = d_correlation(radios, curva, min_puntos_ajuste, d_r2_minimo)

    log.info(
        "Mc=%.2f (%s) | b=%.3f±%.3f sobre %d eventos | d=%.3f (R²=%.4f) "
        "entre %.1f y %.1f km",
        mc,
        mc_metodo,
        b,
        b_sigma,
        n_b,
        d,
        d_r2,
        *d_rango,
    )

    return {
        "mc": mc,
        "mc_metodo": mc_metodo,
        "mc_maxc": maxc,
        "mc_gft": gft,
        "gft_bondad": gft_r,
        "b": b,
        "b_sigma": b_sigma,
        "a": a,
        "d": d,
        "d_r2": d_r2,
        "d_rango_km": list(d_rango),
        "n_eventos_catalogo": int(len(sismos)),
        "n_eventos_completos": n_b,
        "knobs": {
            "mc_metodo": mc_metodo,
            "bin_magnitud": bin_magnitud,
            "correccion_maxc": correccion_maxc,
            "gft_objetivo": gft_objetivo,
            "min_eventos": min_eventos,
            "n_radios": n_radios,
            "min_puntos_ajuste": min_puntos_ajuste,
            "d_r2_minimo": d_r2_minimo,
        },
        # Las curvas van al json para poder graficarlas después sin volver a
        # recorrer los pares, que es la parte cara.
        "curva_fmd": {
            "magnitud": centros.tolist(),
            "no_acumulada": no_acumulada.tolist(),
            "acumulada": acumulada.tolist(),
        },
        "curva_correlacion": {
            "radio_km": radios.tolist(),
            "c": curva.tolist(),
        },
    }
