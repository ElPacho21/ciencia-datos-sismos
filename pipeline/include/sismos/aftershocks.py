"""Thinning probabilístico y catálogo randomizado de contraste.

El umbral eta_0 parte la población con una línea dura, pero esa línea miente en
los bordes: un evento que cae justo por debajo no es más réplica que uno que
cae justo por encima. El thinning reemplaza la línea por un sorteo. A cada
evento se le calcula la probabilidad de ser fondo y se lo devuelve al fondo con
esa probabilidad; los que quedan lejos del umbral casi no cambian de lado y los
del medio se reparten según cuánto los reclama cada población.

Esa probabilidad sale de la **mezcla de dos gaussianas ya ajustada** en
`threshold.py`: es la responsabilidad posterior de la componente del fondo. Por
eso el paso es probabilístico y por eso la semilla es un parámetro del DAG y no
un detalle — sin ella los números del informe no se reproducen.

## Por qué el catálogo randomizado no fija la fracción de fondo

Zaliapin & Ben-Zion estiman el peso del fondo comparando la distribución
observada contra la de un catálogo barajado. Acá el nulo se construye igual
—se permutan tiempos y magnitudes y se dejan los epicentros donde están, para
destruir la asociación temporal conservando la geografía— pero **no** se lo usa
para estimar ese peso, porque sobre este catálogo no lo identifica:

El barajado conserva N. Si de esos N eventos el fondo verdadero es la mitad, el
catálogo barajado tiene el doble de densidad que el fondo que quiere modelar,
sus vecinos caen más cerca y su eta se corre hacia abajo. Sobre un control con
50% de réplicas plantadas, el cociente entre distribuciones devuelve un peso de
fondo de 0.96 en vez de 0.50; submuestrear el nulo para igualar densidades no
arregla nada porque la ecuación de punto fijo o es degenerada (todo peso es
solución) o converge igual al valor equivocado.

Lo que el nulo **sí** contesta, y para lo que se usa acá, es si la bimodalidad
es real o la fabrica la métrica: `contrast_with_null` mide si el modo apretado
sobrevive cuando se destruye toda la estructura temporal. Sobre un catálogo sin
réplicas plantadas, observado y barajado dan idéntico; con réplicas, no.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, partition
from sismos.parameters import MAGNITUDE_BIN, trim_to_mc
from sismos.threshold import fit_mixture, mode_separation, posterior_background
from sismos.neighbors import nearest_neighbor

log = logging.getLogger(__name__)

NULL_DIR = OUTPUT_DIR / "null"
AFTERSHOCKS_DIR = OUTPUT_DIR / "aftershocks"

# Cuántos puntos porcentuales tiene que superar el catálogo real al barajado,
# por debajo del umbral, para que valga la pena creerle a la clasificación.
MIN_EXCESS = 0.05


def null_path(seed, n_repetitions, **query) -> Path:
    """El nulo depende de la semilla y de cuántas veces se barajó.

    Los dos van en el nombre y no adentro del archivo: así dos corridas con
    distinta semilla conviven en disco y se ve de un vistazo cuál es cuál.
    """
    return NULL_DIR / f"{partition(**query)}_seed={seed}_reps={n_repetitions}.parquet"


def aftershocks_path(seed, **query) -> Path:
    return AFTERSHOCKS_DIR / f"{partition(**query)}_seed={seed}.parquet"


def null_write(destination: Path, df: pd.DataFrame) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def null_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def aftershocks_write(destination: Path, df: pd.DataFrame) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def aftershocks_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _shuffle(catalog: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rompe la asociación tiempo-espacio conservando cada distribución marginal.

    Tiempos y magnitudes se permutan por separado; los epicentros se quedan
    donde estaban. Si además se movieran los epicentros, el nulo tendría otra
    dimensión fractal que la del catálogo real y la comparación dejaría de ser
    contra "el mismo catálogo sin réplicas".
    """
    shuffled = catalog.copy()
    n = len(catalog)
    shuffled["time"] = catalog["time"].to_numpy()[rng.permutation(n)]
    shuffled["mag"] = catalog["mag"].to_numpy()[rng.permutation(n)]
    return shuffled.sort_values("time").reset_index(drop=True)


def randomize_catalog(
    quakes: pd.DataFrame,
    b: float,
    d: float,
    mc: float,
    n_repetitions: int = 5,
    seed: int = 0,
    magnitude_bin: float = MAGNITUDE_BIN,
) -> pd.DataFrame:
    """Distribución nula de log10(eta): el catálogo barajado, varias veces.

    Cada repetición es una corrida completa del vecino más cercano, o sea O(N²).
    Con pocas repeticiones el nulo queda ruidoso; con muchas, el paso se vuelve
    la parte cara del pipeline. Cinco alcanza para catálogos de unos pocos miles
    de eventos.
    """
    if n_repetitions < 1:
        raise ValueError("At least one repetition is needed to build the null.")

    catalog = trim_to_mc(quakes, mc, magnitude_bin).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    chunks = []
    for repetition in range(n_repetitions):
        shuffled = nearest_neighbor(
            _shuffle(catalog, rng), b=b, d=d, mc=mc, magnitude_bin=magnitude_bin
        )
        values = shuffled["log10_eta"].dropna().to_numpy()
        chunks.append(
            pd.DataFrame({"log10_eta": values, "repetition": repetition})
        )

    null = pd.concat(chunks, ignore_index=True)

    log.info(
        "Null: %d log10(eta) values over %d shuffles (seed %d), "
        "between %.2f and %.2f (median %.2f).",
        len(null),
        n_repetitions,
        seed,
        null["log10_eta"].min(),
        null["log10_eta"].max(),
        null["log10_eta"].median(),
    )

    return null


def contrast_with_null(
    paired: pd.DataFrame, null: pd.DataFrame, log10_eta0: float
) -> dict:
    """¿La bimodalidad es del catálogo o la fabrica la métrica?

    Le ajusta al nulo la misma mezcla de dos gaussianas que al observado y
    compara. Si el catálogo barajado también se parte en dos modos igual de
    separados, y cae bajo el umbral la misma proporción de eventos, entonces lo
    que se está midiendo no son réplicas: es la forma que tiene eta cuando no
    pasa nada.
    """
    observed = paired["log10_eta"].dropna().to_numpy(dtype=float)
    shuffled = null["log10_eta"].to_numpy(dtype=float)

    _, mu, sigma, _, _ = fit_mixture(shuffled)
    null_separation = mode_separation(mu, sigma)

    fraction_obs = float(np.mean(observed < log10_eta0))
    fraction_null = float(np.mean(shuffled < log10_eta0))
    excess = fraction_obs - fraction_null

    contrast = {
        "fraction_below_threshold_observed": fraction_obs,
        "fraction_below_threshold_null": fraction_null,
        "excess_over_chance": excess,
        "null_mode_separation": null_separation,
        "median_observed": float(np.median(observed)),
        "median_null": float(np.median(shuffled)),
    }

    if excess <= MIN_EXCESS:
        log.warning(
            "Below the threshold fall %.1f%% of the real catalog and %.1f%% of "
            "the shuffled one: the excess over chance is only %.1f points. What "
            "the method is flagging as aftershocks may be just the shape of eta "
            "in a dense catalog.",
            100 * fraction_obs,
            100 * fraction_null,
            100 * excess,
        )
    else:
        log.info(
            "Contrast with the null: below the threshold fall %.1f%% of the real "
            "catalog vs %.1f%% of the shuffled one (excess of %.1f points); "
            "the null separates its modes by %.2f sigma.",
            100 * fraction_obs,
            100 * fraction_null,
            100 * excess,
            null_separation,
        )

    return contrast


def thin(
    paired: pd.DataFrame,
    mixture: dict,
    log10_eta0: float,
    seed: int = 0,
):
    """Devuelve al fondo, por sorteo, los enlaces que el modelo no reclama.

    `parent_id` no se borra nunca: el vecino más cercano de un evento es un
    hecho del catálogo y sirve para auditar aunque el enlace se rechace. Lo que
    dice si ese enlace se acepta es `is_aftershock`.

    Se guarda también `is_aftershock_eta`, la clasificación dura por el umbral,
    para poder comparar las dos y ver cuánto movió el thinning.
    """
    x = paired["log10_eta"].to_numpy(dtype=float)
    has_parent = np.isfinite(x)

    # Un evento sin padre es fondo por definición: no hay enlace que sortear.
    p_background = np.ones(len(x))
    p_background[has_parent] = posterior_background(x[has_parent], mixture)

    rng = np.random.default_rng(seed)
    is_after = has_parent & (rng.random(len(x)) >= p_background)

    by_threshold = has_parent & (x < log10_eta0)

    out = paired.copy()
    out["p_background"] = p_background
    out["is_aftershock"] = is_after
    out["is_aftershock_eta"] = by_threshold

    returned = int((by_threshold & ~is_after).sum())
    added = int((~by_threshold & is_after).sum())

    diagnostics = {
        "seed": int(seed),
        "n_events": int(len(out)),
        "n_aftershocks": int(is_after.sum()),
        "n_aftershocks_by_threshold": int(by_threshold.sum()),
        "n_returned_to_background": returned,
        "n_added_to_aftershocks": added,
        "fraction_aftershocks": float(is_after.mean()),
    }

    log.info(
        "Thinning (seed %d): %d aftershocks out of %d events (%.1f%%) | vs the "
        "hard cut at eta_0: %d returned to background, %d added.",
        seed,
        diagnostics["n_aftershocks"],
        diagnostics["n_events"],
        100 * diagnostics["fraction_aftershocks"],
        returned,
        added,
    )

    return out, diagnostics
