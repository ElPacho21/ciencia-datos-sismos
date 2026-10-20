"""Exploración de regiones candidatas contra el endpoint `count` del USGS.

No usa Airflow ni baja catálogos: sólo cuenta. Sirve para decidir, con números,
la ventana y el rectángulo de cada región antes de correr el pipeline:

- `mc`:      Mc por máxima curvatura (+0,2) en bloques de 5 años desde 1973, y
             cuántos eventos M >= 4,5 y M >= 6 hay en cada bloque.
- `density`: epicentros M >= 4,5 por franja de 1 grado a lo largo del arco, más
             allá de los bordes propuestos, para ver si cada borde cae en un gap.
- `mega`:    los sismos M >= 8 de cada región (Maule, Illapel, Iquique, ...).

Uso:
    python scripts/explore_regions.py --steps mc,density,mega --cache cache.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"
QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

MAGNITUDE_BIN = 0.1
MAXC_CORRECTION = 0.2
ENDTIME = "2026-01-01"
BLOCK_YEARS = 5
FIRST_YEAR = 1973

# (min_lat, max_lat, min_lon, max_lon) y el eje a lo largo del cual corre el arco.
REGIONS = {
    "andes": {"bbox": (-55, -21, -74, -53), "axis": "lat"},
    "andes_north": {"bbox": (-55, -18, -76, -53), "axis": "lat"},
    "japan": {"bbox": (30, 46, 128, 148), "axis": "lat"},
    "philippines": {"bbox": (5, 20, 117, 128), "axis": "lat"},
    "sumatra_java": {"bbox": (-11, 6, 92, 112), "axis": "lon"},
    "mexico_central_america": {"bbox": (8, 21, -108, -83), "axis": "lon"},
}

_cache: dict[str, int] = {}
_cache_path: Path | None = None


def count(starttime, endtime, minmag, maxmag=None, bbox=None) -> int:
    params = {
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": minmag,
        "eventtype": "earthquake",
        "format": "geojson",
    }
    if maxmag is not None:
        params["maxmagnitude"] = maxmag
    if bbox is not None:
        params.update(
            minlatitude=bbox[0], maxlatitude=bbox[1],
            minlongitude=bbox[2], maxlongitude=bbox[3],
        )
    url = COUNT_URL + "?" + urllib.parse.urlencode(params)
    if url in _cache:
        return _cache[url]

    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                value = int(json.load(response)["count"])
            _cache[url] = value
            if len(_cache) % 40 == 0:
                save_cache()
            return value
        except Exception as error:
            if attempt == 7:
                raise
            # 429: el servicio pide ir más despacio; se espera más cada vez.
            slow = "429" in str(error)
            time.sleep((15 if slow else 2) * (attempt + 1))


def save_cache() -> None:
    if _cache_path is not None:
        _cache_path.write_text(json.dumps(_cache), encoding="utf-8")


def blocks():
    years = list(range(FIRST_YEAR, int(ENDTIME[:4]), BLOCK_YEARS))
    for i, year in enumerate(years):
        end = f"{years[i + 1]}-01-01" if i + 1 < len(years) else ENDTIME
        yield f"{year}-01-01", end


def maxc(bin_counts: dict[float, int]) -> float | None:
    if not any(bin_counts.values()):
        return None
    center = max(bin_counts, key=lambda m: (bin_counts[m], -m))
    return round(center + MAXC_CORRECTION, 2)


def mc_table(name: str, bbox) -> list[dict]:
    centers = [round(3.5 + MAGNITUDE_BIN * i, 1) for i in range(26)]  # 3.5 .. 6.0
    jobs = [
        (start, end, m) for start, end in blocks() for m in centers
    ]

    def one(job):
        start, end, m = job
        return count(start, end, round(m - 0.05, 2), round(m + 0.0499, 4), bbox)

    with ThreadPoolExecutor(3) as pool:
        values = list(pool.map(one, jobs))
    by_job = dict(zip(jobs, values))

    rows = []
    for start, end in blocks():
        bins = {m: by_job[(start, end, m)] for m in centers}
        n45 = count(start, end, 4.45, None, bbox)
        n6 = count(start, end, 6.0, None, bbox)
        rows.append(
            {"start": start, "end": end, "mc": maxc(bins), "n_ge_4.5": n45, "n_ge_6": n6}
        )
    return rows


def density(name: str, spec: dict, start="2000-01-01", margin=4) -> list[tuple[int, int]]:
    lat0, lat1, lon0, lon1 = spec["bbox"]
    axis = spec["axis"]
    lo, hi = (lat0, lat1) if axis == "lat" else (lon0, lon1)
    edges = list(range(lo - margin, hi + margin))

    def one(edge):
        if axis == "lat":
            box = (edge, edge + 1, lon0, lon1)
        else:
            box = (lat0, lat1, edge, edge + 1)
        return count(start, ENDTIME, 4.45, None, box)

    with ThreadPoolExecutor(3) as pool:
        values = list(pool.map(one, edges))
    return list(zip(edges, values))


def mega(name: str, bbox) -> list[dict]:
    """Los M >= 8 desde 1973, buscados por conteo anual y detalle del año."""
    events = []
    params = {
        "starttime": f"{FIRST_YEAR}-01-01",
        "endtime": ENDTIME,
        "minmagnitude": 8.0,
        "eventtype": "earthquake",
        "format": "csv",
        "orderby": "time-asc",
        "minlatitude": bbox[0], "maxlatitude": bbox[1],
        "minlongitude": bbox[2], "maxlongitude": bbox[3],
    }
    url = QUERY_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        lines = response.read().decode("utf-8").strip().splitlines()
    header = lines[0].split(",")
    for line in lines[1:]:
        # `place` puede traer comas entre comillas: sólo hacen falta los primeros campos.
        cells = line.split(",")
        row = dict(zip(header, cells))
        events.append(
            {"time": row["time"][:10], "lat": row["latitude"], "lon": row["longitude"], "mag": row["mag"]}
        )
    return events


def main() -> int:
    global _cache_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default="mc,density,mega")
    parser.add_argument("--regions", default=",".join(REGIONS))
    parser.add_argument("--cache", default=None)
    args = parser.parse_args()

    if args.cache:
        _cache_path = Path(args.cache)
        if _cache_path.exists():
            _cache.update(json.loads(_cache_path.read_text(encoding="utf-8")))

    steps = args.steps.split(",")
    for name in args.regions.split(","):
        spec = REGIONS[name]
        print(f"\n=== {name} bbox={spec['bbox']}")
        if "mc" in steps:
            print("block                  Mc   N>=4.5   N>=6")
            for r in mc_table(name, spec["bbox"]):
                print(f"{r['start'][:4]}-{r['end'][:4]}  {str(r['mc']):>8} {r['n_ge_4.5']:>8} {r['n_ge_6']:>6}")
        if "density" in steps:
            print("density along axis (", spec["axis"], ") N>=4.5 since 2000, 1-degree strips:")
            print("  " + " ".join(f"{e}:{n}" for e, n in density(name, spec)))
        if "mega" in steps:
            print("M>=8 since 1973:")
            for e in mega(name, spec["bbox"]):
                print("  ", e)
        save_cache()
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
