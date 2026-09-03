"""Thinning probabilístico y catálogo randomizado de contraste.

El umbral eta_0 parte la población con una línea dura, pero esa línea miente en
los bordes: un evento que cae justo por debajo no es más réplica que uno que
cae justo por encima. El thinning reemplaza la línea por un sorteo. A cada
evento se le calcula la probabilidad de ser fondo y se lo devuelve al fondo con
esa probabilidad; los que quedan lejos del umbral casi no cambian de lado y los
del medio se reparten según cuánto los reclama cada población.

Esa probabilidad sale de la **mezcla de dos gaussianas ya ajustada** en
`umbral.py`: es la responsabilidad posterior de la componente del fondo. Por
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
es real o la fabrica la métrica: `contrastar_con_nulo` mide si el modo apretado
sobrevive cuando se destruye toda la estructura temporal. Sobre un catálogo sin
réplicas plantadas, observado y barajado dan idéntico; con réplicas, no.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, particion
from sismos.parametros import BIN_MAGNITUD, recortar_a_mc
from sismos.umbral import ajustar_mezcla, posterior_fondo, separacion_de_modos
from sismos.vecinos import nearest_neighbor

log = logging.getLogger(__name__)

NULO_DIR = OUTPUT_DIR / "nulo"
REPLICAS_DIR = OUTPUT_DIR / "replicas"

# Cuántos puntos porcentuales tiene que superar el catálogo real al barajado,
# por debajo del umbral, para que valga la pena creerle a la clasificación.
EXCESO_MINIMO = 0.05


def nulo_path(seed, n_repeticiones, **consulta) -> Path:
    """El nulo depende de la semilla y de cuántas veces se barajó.

    Los dos van en el nombre y no adentro del archivo: así dos corridas con
    distinta semilla conviven en disco y se ve de un vistazo cuál es cuál.
    """
    return NULO_DIR / f"{particion(**consulta)}_seed={seed}_reps={n_repeticiones}.parquet"


def replicas_path(seed, **consulta) -> Path:
    return REPLICAS_DIR / f"{particion(**consulta)}_seed={seed}.parquet"


def nulo_write(destino: Path, df: pd.DataFrame) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)


def nulo_read(ruta: Path) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def replicas_write(destino: Path, df: pd.DataFrame) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)


def replicas_read(ruta: Path) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def _barajar(catalogo: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Rompe la asociación tiempo-espacio conservando cada distribución marginal.

    Tiempos y magnitudes se permutan por separado; los epicentros se quedan
    donde estaban. Si además se movieran los epicentros, el nulo tendría otra
    dimensión fractal que la del catálogo real y la comparación dejaría de ser
    contra "el mismo catálogo sin réplicas".
    """
    barajado = catalogo.copy()
    n = len(catalogo)
    barajado["time"] = catalogo["time"].to_numpy()[rng.permutation(n)]
    barajado["mag"] = catalogo["mag"].to_numpy()[rng.permutation(n)]
    return barajado.sort_values("time").reset_index(drop=True)


def randomize_catalog(
    sismos: pd.DataFrame,
    b: float,
    d: float,
    mc: float,
    n_repeticiones: int = 5,
    seed: int = 0,
    bin_magnitud: float = BIN_MAGNITUD,
) -> pd.DataFrame:
    """Distribución nula de log10(eta): el catálogo barajado, varias veces.

    Cada repetición es una corrida completa del vecino más cercano, o sea O(N²).
    Con pocas repeticiones el nulo queda ruidoso; con muchas, el paso se vuelve
    la parte cara del pipeline. Cinco alcanza para catálogos de unos pocos miles
    de eventos.
    """
    if n_repeticiones < 1:
        raise ValueError("Hace falta al menos una repetición para armar el nulo.")

    catalogo = recortar_a_mc(sismos, mc, bin_magnitud).reset_index(drop=True)
    rng = np.random.default_rng(seed)

    pedazos = []
    for repeticion in range(n_repeticiones):
        barajado = nearest_neighbor(
            _barajar(catalogo, rng), b=b, d=d, mc=mc, bin_magnitud=bin_magnitud
        )
        valores = barajado["log10_eta"].dropna().to_numpy()
        pedazos.append(
            pd.DataFrame({"log10_eta": valores, "repeticion": repeticion})
        )

    nulo = pd.concat(pedazos, ignore_index=True)

    log.info(
        "Nulo: %d valores de log10(eta) en %d barajadas (semilla %d), "
        "entre %.2f y %.2f (mediana %.2f).",
        len(nulo),
        n_repeticiones,
        seed,
        nulo["log10_eta"].min(),
        nulo["log10_eta"].max(),
        nulo["log10_eta"].median(),
    )

    return nulo


def contrastar_con_nulo(
    emparentados: pd.DataFrame, nulo: pd.DataFrame, log10_eta0: float
) -> dict:
    """¿La bimodalidad es del catálogo o la fabrica la métrica?

    Le ajusta al nulo la misma mezcla de dos gaussianas que al observado y
    compara. Si el catálogo barajado también se parte en dos modos igual de
    separados, y cae bajo el umbral la misma proporción de eventos, entonces lo
    que se está midiendo no son réplicas: es la forma que tiene eta cuando no
    pasa nada.
    """
    observado = emparentados["log10_eta"].dropna().to_numpy(dtype=float)
    barajado = nulo["log10_eta"].to_numpy(dtype=float)

    _, mu, sigma, _, _ = ajustar_mezcla(barajado)
    separacion_nulo = separacion_de_modos(mu, sigma)

    fraccion_obs = float(np.mean(observado < log10_eta0))
    fraccion_nulo = float(np.mean(barajado < log10_eta0))
    exceso = fraccion_obs - fraccion_nulo

    contraste = {
        "fraccion_bajo_umbral_observado": fraccion_obs,
        "fraccion_bajo_umbral_nulo": fraccion_nulo,
        "exceso_sobre_el_azar": exceso,
        "separacion_modos_nulo": separacion_nulo,
        "mediana_observado": float(np.median(observado)),
        "mediana_nulo": float(np.median(barajado)),
    }

    if exceso <= EXCESO_MINIMO:
        log.warning(
            "Bajo el umbral cae el %.1f%% del catálogo real y el %.1f%% del "
            "barajado: el exceso sobre el azar es de apenas %.1f puntos. Lo que "
            "el método está marcando como réplicas puede ser sólo la forma de "
            "eta en un catálogo denso.",
            100 * fraccion_obs,
            100 * fraccion_nulo,
            100 * exceso,
        )
    else:
        log.info(
            "Contraste con el nulo: bajo el umbral cae el %.1f%% del catálogo "
            "real contra el %.1f%% del barajado (exceso de %.1f puntos); "
            "el nulo separa sus modos %.2f sigma.",
            100 * fraccion_obs,
            100 * fraccion_nulo,
            100 * exceso,
            separacion_nulo,
        )

    return contraste


def thin(
    emparentados: pd.DataFrame,
    mezcla: dict,
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
    x = emparentados["log10_eta"].to_numpy(dtype=float)
    con_padre = np.isfinite(x)

    # Un evento sin padre es fondo por definición: no hay enlace que sortear.
    p_fondo = np.ones(len(x))
    p_fondo[con_padre] = posterior_fondo(x[con_padre], mezcla)

    rng = np.random.default_rng(seed)
    es_replica = con_padre & (rng.random(len(x)) >= p_fondo)

    por_umbral = con_padre & (x < log10_eta0)

    salida = emparentados.copy()
    salida["p_fondo"] = p_fondo
    salida["is_aftershock"] = es_replica
    salida["is_aftershock_eta"] = por_umbral

    devueltos = int((por_umbral & ~es_replica).sum())
    sumados = int((~por_umbral & es_replica).sum())

    diagnostico = {
        "seed": int(seed),
        "n_eventos": int(len(salida)),
        "n_replicas": int(es_replica.sum()),
        "n_replicas_por_umbral": int(por_umbral.sum()),
        "n_devueltos_al_fondo": devueltos,
        "n_sumados_a_replicas": sumados,
        "fraccion_replicas": float(es_replica.mean()),
    }

    log.info(
        "Thinning (semilla %d): %d réplicas de %d eventos (%.1f%%) | contra el "
        "corte duro por eta_0: %d devueltos al fondo, %d sumados.",
        seed,
        diagnostico["n_replicas"],
        diagnostico["n_eventos"],
        100 * diagnostico["fraccion_replicas"],
        devueltos,
        sumados,
    )

    return salida, diagnostico
