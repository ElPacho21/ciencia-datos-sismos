"""Regiones independientes: configuración, efecto borde y concatenación.

El método de Zaliapin & Ben-Zion supone un catálogo homogéneo: la misma
completitud y la misma geometría de fallas. Mezclar regiones en un solo ajuste
lo rompe —el umbral eta_0 queda entre dos poblaciones y los árboles encadenan
miles de km—. Por eso cada región se corre por separado, con su propio Mc y su
propio eta_0, y recién después se concatenan las secuencias resultantes.

Este módulo tiene tres partes:

- la configuración versionada (`regions.yaml`) y su validación;
- el efecto borde: qué tan cerca del límite del rectángulo cae cada sismo
  principal, porque un mainshock pegado al borde pierde las réplicas que caen
  afuera;
- la concatenación de las secuencias de todas las regiones.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos.geo import EARTH_RADIUS_KM

log = logging.getLogger(__name__)

REGIONS_FILE = Path(__file__).with_name("regions.yaml")

# Cuánto se considera "cerca del borde" por defecto. Con 300 km quedaba marcado
# el 60 % de la corrida de referencia (el rectángulo de Andes es angosto) y un
# indicador así no informa; con 100 km queda marcado alrededor del 17 %.
BORDER_BUFFER_KM = 100.0

REQUIRED_KEYS = (
    "name",
    "minlatitude",
    "maxlatitude",
    "minlongitude",
    "maxlongitude",
    "starttime",
    "endtime",
    "minmagnitude",
)

BBOX_KEYS = ("minlatitude", "maxlatitude", "minlongitude", "maxlongitude")


def border_suffix(border_buffer_km=BORDER_BUFFER_KM) -> str:
    """Sufijo de nombre: vacío con el buffer por defecto."""
    number = float(border_buffer_km)
    if number == BORDER_BUFFER_KM:
        return ""
    return f"_border={int(number) if number.is_integer() else number}"


def regions_tag(regions: list[dict]) -> str:
    """Huella corta de la configuración: qué regiones y qué ventanas se combinaron.

    El csv combinado depende de *todas* las regiones; si cambia una bbox o una
    fecha, el contenido es otro y no puede compartir nombre con el anterior.
    """
    canonical = json.dumps(regions, sort_keys=True, default=str)
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    return f"{len(regions)}regions-{digest}"


def load_regions(path: Path = REGIONS_FILE) -> list[dict]:
    """Lee y valida la configuración de regiones."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    regions = raw["regions"] if isinstance(raw, dict) else raw
    validate_regions(regions)
    return regions


def _overlap(a: dict, b: dict) -> bool:
    return not (
        a["maxlatitude"] <= b["minlatitude"]
        or b["maxlatitude"] <= a["minlatitude"]
        or a["maxlongitude"] <= b["minlongitude"]
        or b["maxlongitude"] <= a["minlongitude"]
    )


def validate_regions(regions: list[dict]) -> None:
    """Chequeos que se pagan caro si se descubren después de correr todo."""
    if not regions:
        raise ValueError("The regions list is empty.")

    names = [r.get("name") for r in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"Region names must be unique, got {names}.")

    for r in regions:
        missing = [k for k in REQUIRED_KEYS if k not in r]
        if missing:
            raise ValueError(f"Region {r.get('name')!r} is missing {missing}.")
        if not (-90 <= r["minlatitude"] < r["maxlatitude"] <= 90):
            raise ValueError(f"Region {r['name']!r}: invalid latitude range.")
        # El FDSN no admite minlongitude > maxlongitude, así que ninguna región
        # puede cruzar el antimeridiano.
        if not (-180 <= r["minlongitude"] < r["maxlongitude"] <= 180):
            raise ValueError(
                f"Region {r['name']!r}: invalid longitude range (the service "
                "does not accept a rectangle that crosses the antimeridian)."
            )
        if str(r["starttime"]) >= str(r["endtime"]):
            raise ValueError(f"Region {r['name']!r}: starttime >= endtime.")

    for i, a in enumerate(regions):
        for b in regions[i + 1 :]:
            if _overlap(a, b):
                raise ValueError(
                    f"The cores of {a['name']!r} and {b['name']!r} overlap: an "
                    "event could be a mainshock in two regions."
                )


def distance_to_border_km(lat, lon, bbox) -> np.ndarray:
    """Distancia, sobre la esfera, del epicentro al borde más cercano del rectángulo.

    `bbox` es (min_lat, max_lat, min_lon, max_lon) en grados. Para un borde de
    latitud la distancia es `R · |Δlat|` (en radianes). Para un meridiano es la
    distancia perpendicular al círculo máximo que lo contiene,
    `R · asin(|sin(Δlon) · cos(lat)|)`: a latitudes altas los meridianos
    convergen y un grado de longitud mide mucho menos que en el ecuador.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    min_lat, max_lat, min_lon, max_lon = bbox

    to_south = EARTH_RADIUS_KM * np.abs(np.radians(lat - min_lat))
    to_north = EARTH_RADIUS_KM * np.abs(np.radians(max_lat - lat))

    cos_lat = np.cos(np.radians(lat))

    def meridian(edge):
        s = np.abs(np.sin(np.radians(lon - edge)) * cos_lat)
        return EARTH_RADIUS_KM * np.arcsin(np.clip(s, 0.0, 1.0))

    return np.minimum.reduce([to_south, to_north, meridian(min_lon), meridian(max_lon)])


def add_border_columns(
    summary: pd.DataFrame, bbox, buffer_km: float = BORDER_BUFFER_KM
) -> pd.DataFrame:
    """Agrega `dist_to_border_km` y `near_border` (al final, sin tocar el resto)."""
    out = summary.copy()
    out["dist_to_border_km"] = distance_to_border_km(
        out["mainshock_lat"], out["mainshock_lon"], bbox
    )
    out["near_border"] = out["dist_to_border_km"] < buffer_km
    return out


def combine_regions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatena las secuencias de todas las regiones y agrega `region`.

    Falla si el `cluster_id` o el `mainshock_id` se repiten entre regiones: un
    sismo no puede ser sismo principal en dos regiones, y si pasa los núcleos
    se están solapando o una corrida quedó con datos de otra.
    """
    if not frames:
        raise ValueError("There are no regions to combine.")

    parts = []
    for name, frame in frames.items():
        part = frame.copy()
        part["region"] = name
        parts.append(part)

    # Las columnas tienen que coincidir: una región con features de menos
    # dejaría nulos que no son estructurales.
    columns = list(parts[0].columns)
    for name, part in zip(frames, parts):
        if set(part.columns) != set(columns):
            raise ValueError(
                f"Region {name!r} has different columns: "
                f"{sorted(set(part.columns) ^ set(columns))}."
            )

    combined = pd.concat([p[columns] for p in parts], ignore_index=True)

    for key in ("cluster_id", "mainshock_id"):
        repeated = combined[combined[key].duplicated(keep=False)]
        if len(repeated):
            raise ValueError(
                f"{key!r} repeats across regions "
                f"({repeated.groupby(key)['region'].unique().head(5).to_dict()}): "
                "the cores overlap or a run mixed data."
            )

    log.info(
        "Combined %d regions into %d sequences (%s).",
        len(frames),
        len(combined),
        {n: len(f) for n, f in frames.items()},
    )
    return combined
