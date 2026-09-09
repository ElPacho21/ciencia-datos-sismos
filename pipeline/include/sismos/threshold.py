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

from sismos import OUTPUT_DIR, partition

log = logging.getLogger(__name__)

THRESHOLD_DIR = OUTPUT_DIR / "threshold"

# Por debajo de esto los dos modos están tan encimados que el cruce cae en
# cualquier lado. Es la distancia entre medias en unidades de desvío típico.
MIN_SEPARATION = 1.5

# Una componente con menos peso que esto no es un modo, es ruido que el EM
# acomodó en una esquina.
MIN_WEIGHT = 0.05


def threshold_path(**query) -> Path:
    return THRESHOLD_DIR / f"{partition(**query)}.json"


def threshold_write(destination: Path, threshold: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(threshold, indent=2), encoding="utf-8")


def threshold_read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _density(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def posterior_background(log10_eta, mixture: dict) -> np.ndarray:
    """Probabilidad de que cada evento sea fondo, según la mezcla ajustada.

    Es la misma responsabilidad que calcula el EM, evaluada fuera del ajuste:
    qué fracción de la densidad en ese punto aporta la componente del fondo.

    La diferencia con el corte duro por eta_0 es todo el asunto: un evento
    justo al lado del umbral no cae de un lado ni del otro, queda con una
    probabilidad intermedia. Eso es lo que después sortea el thinning.
    """
    x = np.asarray(log10_eta, dtype=float)
    f_aftershocks = mixture["weight_aftershocks"] * _density(
        x, mixture["mu_aftershocks"], mixture["sigma_aftershocks"]
    )
    f_background = mixture["weight_background"] * _density(
        x, mixture["mu_background"], mixture["sigma_background"]
    )
    return f_background / np.maximum(f_aftershocks + f_background, 1e-300)


def mode_separation(mu, sigma) -> float:
    """Distancia entre medias en unidades de desvío típico."""
    return float((mu[1] - mu[0]) / math.sqrt((sigma[0] ** 2 + sigma[1] ** 2) / 2))


def fit_mixture(x: np.ndarray, max_iter: int = 500, tol: float = 1e-9):
    """EM para una mezcla de dos gaussianas en una dimensión.

    Arranca con las medias en los cuartiles en vez de al azar: el resultado es
    reproducible sin necesidad de semilla, que para un umbral que después hay
    que defender en el informe importa más que ganar una iteración.

    Devuelve las componentes ordenadas por media, así la 0 es siempre la de las
    réplicas y la 1 la del fondo.
    """
    if len(x) < 20:
        raise ValueError(f"Only {len(x)} eta values: not enough to fit.")

    mu = np.array([np.percentile(x, 25), np.percentile(x, 75)], dtype=float)
    sigma = np.full(2, max(float(x.std()) / 2, 1e-3))
    weights = np.array([0.5, 0.5])

    log_likelihood = -np.inf
    iterations = 0

    for iterations in range(1, max_iter + 1):
        densities = weights * np.column_stack(
            [_density(x, mu[k], sigma[k]) for k in range(2)]
        )
        total = densities.sum(axis=1)
        # Un punto lejísimos de las dos componentes da densidad 0 y rompería la
        # división; el piso lo deja con responsabilidad repartida.
        total = np.maximum(total, 1e-300)

        responsibility = densities / total[:, None]
        new_ll = float(np.log(total).sum())

        n_k = responsibility.sum(axis=0)
        weights = n_k / len(x)
        mu = (responsibility * x[:, None]).sum(axis=0) / n_k
        sigma = np.sqrt(
            (responsibility * (x[:, None] - mu) ** 2).sum(axis=0) / n_k
        )
        sigma = np.maximum(sigma, 1e-6)

        if abs(new_ll - log_likelihood) < tol:
            log_likelihood = new_ll
            break
        log_likelihood = new_ll

    order = np.argsort(mu)
    return (
        weights[order],
        mu[order],
        sigma[order],
        log_likelihood,
        iterations,
    )


def component_crossing(weights, mu, sigma, n_grid: int = 20001):
    """Punto entre las dos medias donde las componentes pesadas se igualan.

    Es el criterio de mínimo error de clasificación: a la izquierda del cruce
    manda la componente de réplicas, a la derecha la del fondo. Se busca sobre
    una grilla y se interpola en vez de resolver la cuadrática, que con sigmas
    parecidas se vuelve numéricamente incómoda.
    """
    grid = np.linspace(mu[0], mu[1], n_grid)
    difference = weights[0] * _density(grid, mu[0], sigma[0]) - weights[1] * _density(
        grid, mu[1], sigma[1]
    )

    changes = np.where(np.diff(np.sign(difference)) != 0)[0]
    if len(changes) == 0:
        return None

    i = int(changes[0])
    x0, x1 = grid[i], grid[i + 1]
    y0, y1 = difference[i], difference[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def fit_threshold(
    paired: pd.DataFrame,
    manual_log10_eta0: float | None = None,
    n_histogram_bins: int = 60,
) -> dict:
    """Ajusta la mezcla sobre log10(eta) y devuelve eta_0 con sus diagnósticos.

    `manual_log10_eta0` pisa el valor ajustado pero **no** saltea el ajuste: la
    mezcla se calcula igual para poder comparar el número puesto a mano contra
    lo que dicen los datos.
    """
    x = paired["log10_eta"].dropna().to_numpy(dtype=float)

    weights, mu, sigma, log_likelihood, iterations = fit_mixture(x)

    # Distancia entre medias en unidades de desvío: es lo que dice si los dos
    # modos son de verdad dos modos o un solo bulto que el EM partió al medio.
    separation = mode_separation(mu, sigma)

    fitted = component_crossing(weights, mu, sigma)

    if fitted is None:
        log.warning(
            "The two components do not cross between their means: the midpoint "
            "is used as the threshold, but the fit is not describing two modes."
        )
        fitted = float(mu.mean())

    log10_eta0 = manual_log10_eta0 if manual_log10_eta0 is not None else fitted

    if float(weights.min()) < MIN_WEIGHT:
        log.warning(
            "One of the components kept only %.1f%% of the data: EM did not "
            "find two real populations, it found one and a corner.",
            100 * float(weights.min()),
        )

    if separation < MIN_SEPARATION:
        log.warning(
            "The modes are only %.2f std apart (reasonable minimum %.1f): "
            "eta_0=%.3f is fragile; set it by hand or narrow the region.",
            separation,
            MIN_SEPARATION,
            log10_eta0,
        )

    # Error de clasificación esperado si se corta en el umbral: cuánta réplica
    # queda del lado del fondo y viceversa, según el propio modelo ajustado.
    false_background = weights[0] * (1 - _normal_cdf(log10_eta0, mu[0], sigma[0]))
    false_aftershocks = weights[1] * _normal_cdf(log10_eta0, mu[1], sigma[1])

    n_aftershocks = int((x < log10_eta0).sum())

    counts, edges = np.histogram(x, bins=n_histogram_bins)

    log.info(
        "eta_0: log10=%.3f%s | modes at %.2f and %.2f (separation %.2f sigma, "
        "weights %.2f/%.2f) | %d of %d events below (%.1f%%) | "
        "expected classification error %.1f%%",
        log10_eta0,
        " (set by hand)" if manual_log10_eta0 is not None else "",
        mu[0],
        mu[1],
        separation,
        weights[0],
        weights[1],
        n_aftershocks,
        len(x),
        100 * n_aftershocks / len(x),
        100 * (false_background + false_aftershocks),
    )

    return {
        "log10_eta0": float(log10_eta0),
        "eta0": float(10**log10_eta0),
        "log10_eta0_fitted": float(fitted),
        "set_manually": manual_log10_eta0 is not None,
        "mixture": {
            "weight_aftershocks": float(weights[0]),
            "weight_background": float(weights[1]),
            "mu_aftershocks": float(mu[0]),
            "mu_background": float(mu[1]),
            "sigma_aftershocks": float(sigma[0]),
            "sigma_background": float(sigma[1]),
            "separation_sigmas": separation,
            "log_likelihood": float(log_likelihood),
            "iterations": int(iterations),
        },
        "n_events": int(len(x)),
        "n_below_threshold": n_aftershocks,
        "fraction_below_threshold": float(n_aftershocks / len(x)),
        "expected_classification_error": float(false_background + false_aftershocks),
        # Para poder graficar el histograma con las dos curvas encima sin
        # volver a leer el parquet de vecinos.
        "histogram": {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
        },
    }
