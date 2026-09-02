"""Umbral eta_0 que separa las réplicas del fondo.

Zaliapin & Ben-Zion observan que la distribución de log10(eta) de un catálogo
real es bimodal: un modo a valores bajos (pares apretados en tiempo y espacio,
las réplicas) y otro a valores altos (pares sueltos, el fondo independiente).
El umbral no se fija a ojo, se ajusta: se le pone una mezcla de dos gaussianas
a log10(eta) y eta_0 queda donde las dos componentes se cruzan.

El ajuste es por EM escrito a mano en vez de traer scikit-learn. Son treinta
líneas para una mezcla de dos gaussianas en una dimensión, y así queda a la
vista qué se está haciendo en lugar de esconderlo atrás de una librería.

Cuidado: el método presupone que la bimodalidad *existe*. Sobre un catálogo
chico, o global con Mc heterogéneo, los dos modos se pisan y eta_0 pasa a ser
un número frágil. Por eso acá se reportan la separación entre modos y el error
de clasificación esperado, y se avisa cuando el ajuste no se sostiene.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, particion

log = logging.getLogger(__name__)

UMBRAL_DIR = OUTPUT_DIR / "umbral"

# Por debajo de esto los dos modos están tan encimados que el cruce cae en
# cualquier lado. Es la distancia entre medias en unidades de desvío típico.
SEPARACION_MINIMA = 1.5

# Una componente con menos peso que esto no es un modo, es ruido que el EM
# acomodó en una esquina.
PESO_MINIMO = 0.05


def umbral_path(starttime, endtime, minmagnitude) -> Path:
    return UMBRAL_DIR / f"{particion(starttime, endtime, minmagnitude)}.json"


def umbral_write(destino: Path, umbral: dict) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(umbral, indent=2), encoding="utf-8")


def umbral_read(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _densidad(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _normal_acumulada(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def posterior_fondo(log10_eta, mezcla: dict) -> np.ndarray:
    """Probabilidad de que cada evento sea fondo, según la mezcla ajustada.

    Es la misma responsabilidad que calcula el EM, evaluada fuera del ajuste:
    qué fracción de la densidad en ese punto aporta la componente del fondo.

    La diferencia con el corte duro por eta_0 es todo el asunto: un evento
    justo al lado del umbral no cae de un lado ni del otro, queda con una
    probabilidad intermedia. Eso es lo que después sortea el thinning.
    """
    x = np.asarray(log10_eta, dtype=float)
    f_replicas = mezcla["peso_replicas"] * _densidad(
        x, mezcla["mu_replicas"], mezcla["sigma_replicas"]
    )
    f_fondo = mezcla["peso_fondo"] * _densidad(
        x, mezcla["mu_fondo"], mezcla["sigma_fondo"]
    )
    return f_fondo / np.maximum(f_replicas + f_fondo, 1e-300)


def separacion_de_modos(mu, sigma) -> float:
    """Distancia entre medias en unidades de desvío típico."""
    return float((mu[1] - mu[0]) / math.sqrt((sigma[0] ** 2 + sigma[1] ** 2) / 2))


def ajustar_mezcla(x: np.ndarray, max_iter: int = 500, tol: float = 1e-9):
    """EM para una mezcla de dos gaussianas en una dimensión.

    Arranca con las medias en los cuartiles en vez de al azar: el resultado es
    reproducible sin necesidad de semilla, que para un umbral que después hay
    que defender en el informe importa más que ganar una iteración.

    Devuelve las componentes ordenadas por media, así la 0 es siempre la de las
    réplicas y la 1 la del fondo.
    """
    if len(x) < 20:
        raise ValueError(f"Sólo {len(x)} valores de eta: no alcanza para ajustar.")

    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)], dtype=float)
    sigma = np.full(2, max(float(x.std()) / 2, 1e-3))
    pesos = np.array([0.5, 0.5])

    verosimilitud = -np.inf
    iteraciones = 0

    for iteraciones in range(1, max_iter + 1):
        densidades = pesos * np.column_stack(
            [_densidad(x, mu[k], sigma[k]) for k in range(2)]
        )
        total = densidades.sum(axis=1)
        # Un punto lejísimos de las dos componentes da densidad 0 y rompería la
        # división; el piso lo deja con responsabilidad repartida.
        total = np.maximum(total, 1e-300)

        responsabilidad = densidades / total[:, None]
        nueva = float(np.log(total).sum())

        n_k = responsabilidad.sum(axis=0)
        pesos = n_k / len(x)
        mu = (responsabilidad * x[:, None]).sum(axis=0) / n_k
        sigma = np.sqrt(
            (responsabilidad * (x[:, None] - mu) ** 2).sum(axis=0) / n_k
        )
        sigma = np.maximum(sigma, 1e-6)

        if abs(nueva - verosimilitud) < tol:
            verosimilitud = nueva
            break
        verosimilitud = nueva

    orden = np.argsort(mu)
    return (
        pesos[orden],
        mu[orden],
        sigma[orden],
        verosimilitud,
        iteraciones,
    )


def cruce_componentes(pesos, mu, sigma, n_grilla: int = 20001):
    """Punto entre las dos medias donde las componentes pesadas se igualan.

    Es el criterio de mínimo error de clasificación: a la izquierda del cruce
    manda la componente de réplicas, a la derecha la del fondo. Se busca sobre
    una grilla y se interpola en vez de resolver la cuadrática, que con sigmas
    parecidas se vuelve numéricamente incómoda.
    """
    grilla = np.linspace(mu[0], mu[1], n_grilla)
    diferencia = pesos[0] * _densidad(grilla, mu[0], sigma[0]) - pesos[1] * _densidad(
        grilla, mu[1], sigma[1]
    )

    cambios = np.where(np.diff(np.sign(diferencia)) != 0)[0]
    if len(cambios) == 0:
        return None

    i = int(cambios[0])
    x0, x1 = grilla[i], grilla[i + 1]
    y0, y1 = diferencia[i], diferencia[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def fit_threshold(
    emparentados: pd.DataFrame,
    log10_eta0_manual: float | None = None,
    n_bins_histograma: int = 60,
) -> dict:
    """Ajusta la mezcla sobre log10(eta) y devuelve eta_0 con sus diagnósticos.

    `log10_eta0_manual` pisa el valor ajustado pero **no** saltea el ajuste: la
    mezcla se calcula igual para poder comparar el número puesto a mano contra
    lo que dicen los datos.
    """
    x = emparentados["log10_eta"].dropna().to_numpy(dtype=float)

    pesos, mu, sigma, verosimilitud, iteraciones = ajustar_mezcla(x)

    # Distancia entre medias en unidades de desvío: es lo que dice si los dos
    # modos son de verdad dos modos o un solo bulto que el EM partió al medio.
    separacion = separacion_de_modos(mu, sigma)

    ajustado = cruce_componentes(pesos, mu, sigma)

    if ajustado is None:
        log.warning(
            "Las dos componentes no se cruzan entre sus medias: se usa el punto "
            "medio como umbral, pero el ajuste no está describiendo dos modos."
        )
        ajustado = float(mu.mean())

    log10_eta0 = log10_eta0_manual if log10_eta0_manual is not None else ajustado

    if float(pesos.min()) < PESO_MINIMO:
        log.warning(
            "Una de las componentes se quedó con el %.1f%% de los datos: el EM "
            "no encontró dos poblaciones reales, encontró una y un rincón.",
            100 * float(pesos.min()),
        )

    if separacion < SEPARACION_MINIMA:
        log.warning(
            "Los modos están separados sólo %.2f desvíos (mínimo razonable %.1f): "
            "eta_0=%.3f es frágil y conviene fijarlo a mano o acotar la región.",
            separacion,
            SEPARACION_MINIMA,
            log10_eta0,
        )

    # Error de clasificación esperado si se corta en el umbral: cuánta réplica
    # queda del lado del fondo y viceversa, según el propio modelo ajustado.
    falsos_fondo = pesos[0] * (1 - _normal_acumulada(log10_eta0, mu[0], sigma[0]))
    falsas_replicas = pesos[1] * _normal_acumulada(log10_eta0, mu[1], sigma[1])

    n_replicas = int((x < log10_eta0).sum())

    conteos, bordes = np.histogram(x, bins=n_bins_histograma)

    log.info(
        "eta_0: log10=%.3f%s | modos en %.2f y %.2f (separación %.2f sigma, "
        "pesos %.2f/%.2f) | %d de %d eventos por debajo (%.1f%%) | "
        "error de clasificación esperado %.1f%%",
        log10_eta0,
        " (fijado a mano)" if log10_eta0_manual is not None else "",
        mu[0],
        mu[1],
        separacion,
        pesos[0],
        pesos[1],
        n_replicas,
        len(x),
        100 * n_replicas / len(x),
        100 * (falsos_fondo + falsas_replicas),
    )

    return {
        "log10_eta0": float(log10_eta0),
        "eta0": float(10**log10_eta0),
        "log10_eta0_ajustado": float(ajustado),
        "fijado_a_mano": log10_eta0_manual is not None,
        "mezcla": {
            "peso_replicas": float(pesos[0]),
            "peso_fondo": float(pesos[1]),
            "mu_replicas": float(mu[0]),
            "mu_fondo": float(mu[1]),
            "sigma_replicas": float(sigma[0]),
            "sigma_fondo": float(sigma[1]),
            "separacion_sigmas": separacion,
            "log_verosimilitud": float(verosimilitud),
            "iteraciones": int(iteraciones),
        },
        "n_eventos": int(len(x)),
        "n_bajo_umbral": n_replicas,
        "fraccion_bajo_umbral": float(n_replicas / len(x)),
        "error_clasificacion_esperado": float(falsos_fondo + falsas_replicas),
        # Para poder graficar el histograma con las dos curvas encima sin
        # volver a leer el parquet de vecinos.
        "histograma": {
            "bordes": bordes.tolist(),
            "conteos": conteos.tolist(),
        },
    }
