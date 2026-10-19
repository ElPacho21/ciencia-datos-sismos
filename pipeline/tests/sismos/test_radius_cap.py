"""Tests del tope de distancia padre-hijo (r_max_km)."""

import numpy as np
import pandas as pd
import pytest

from sismos import method_suffix
from sismos.aftershocks import randomize_catalog
from sismos.neighbors import gardner_knopoff_km, nearest_neighbor

B, D, MC = 1.0, 1.5, 4.0


def random_catalog(n=400, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01", tz="UTC")
    catalog = pd.DataFrame(
        {
            "id": [f"e{i}" for i in range(n)],
            "time": start + pd.to_timedelta(np.sort(rng.uniform(0, 3e7, n)), unit="s"),
            "latitude": rng.uniform(-40, -20, n),
            "longitude": rng.uniform(-75, -60, n),
            "depth": 10.0,
            "mag": 4.0 + rng.exponential(0.5, n),
        }
    )
    return catalog.sort_values("time").reset_index(drop=True)


def two_clusters():
    """Un grupo cerca de (-30, -70) y otro a ~1.100 km, en el mismo catálogo."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
    rows = [
        ("A", 0, -30.0, -70.0, 7.0),
        ("B", 5, -30.1, -70.1, 4.5),  # a ~15 km de A
        ("C", 10, -40.0, -70.0, 4.5),  # a ~1.100 km de A: sólo A está antes que él... y B
    ]
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "time": [start + pd.Timedelta(hours=r[1]) for r in rows],
            "latitude": [r[2] for r in rows],
            "longitude": [r[3] for r in rows],
            "depth": 10.0,
            "mag": [r[4] for r in rows],
        }
    )


@pytest.mark.parametrize("kwargs", [{"r_max_km": 1e6}, {"r_max_mode": "gk", "r_max_k": 1e6}])
def test_radio_enorme_da_exactamente_los_mismos_padres_y_eta(kwargs):
    catalog = random_catalog()
    reference = nearest_neighbor(catalog, b=B, d=D, mc=MC)
    capped = nearest_neighbor(catalog, b=B, d=D, mc=MC, **kwargs)

    assert (reference["parent_id"].fillna("") == capped["parent_id"].fillna("")).all()
    pd.testing.assert_frame_equal(
        reference.drop(columns="parent_id"), capped.drop(columns="parent_id")
    )


def test_sin_tope_el_camino_es_el_de_siempre():
    catalog = random_catalog()
    a = nearest_neighbor(catalog, b=B, d=D, mc=MC)
    b = nearest_neighbor(catalog, b=B, d=D, mc=MC, r_max_km=None, r_max_mode="fixed")
    pd.testing.assert_frame_equal(a, b)


def test_un_par_mas_lejos_que_el_radio_no_puede_ser_padre_hijo():
    paired = nearest_neighbor(two_clusters(), b=B, d=D, mc=MC, r_max_km=300).set_index("id")

    assert paired.loc["B", "parent_id"] == "A"
    # C queda a ~1.100 km de A y B: sin candidato dentro del radio es raíz.
    assert pd.isna(paired.loc["C", "parent_id"])
    assert np.isnan(paired.loc["C", "eta"])


def test_sin_tope_el_lejano_si_reclama_padre():
    paired = nearest_neighbor(two_clusters(), b=B, d=D, mc=MC).set_index("id")
    assert paired.loc["C", "parent_id"] == "A"


def test_el_tope_elige_otro_padre_dentro_del_radio():
    # Sin tope gana A (M8, a ~100 km de D) porque su magnitud compensa la
    # distancia y la espera; con tope de 50 km A queda afuera y el padre pasa a
    # ser B, que está a ~11 km.
    start = pd.Timestamp("2020-01-01", tz="UTC")
    rows = [
        ("A", 0, -33.1, -70.0, 8.0),
        ("B", 1, -33.9, -70.0, 4.5),
        ("D", 2001, -34.0, -70.0, 4.5),
    ]
    catalog = pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "time": [start + pd.Timedelta(hours=r[1]) for r in rows],
            "latitude": [r[2] for r in rows],
            "longitude": [r[3] for r in rows],
            "depth": 10.0,
            "mag": [r[4] for r in rows],
        }
    )
    free = nearest_neighbor(catalog, b=B, d=D, mc=MC).set_index("id")
    capped = nearest_neighbor(catalog, b=B, d=D, mc=MC, r_max_km=50).set_index("id")

    assert free.loc["D", "parent_id"] == "A"
    assert capped.loc["D", "parent_id"] == "B"


def test_modo_gk_el_radio_depende_de_la_magnitud_del_padre():
    # Gardner-Knopoff: M4.5 -> ~14 km * 3; M7.0 -> ~72 km * 3 = ~216 km.
    assert gardner_knopoff_km(4.5) == pytest.approx(10 ** (0.1238 * 4.5 + 0.983))
    start = pd.Timestamp("2020-01-01", tz="UTC")

    def catalog(parent_mag):
        # hijo a ~110 km (1 grado de latitud) del padre
        return pd.DataFrame(
            {
                "id": ["P", "H"],
                "time": [start, start + pd.Timedelta(hours=1)],
                "latitude": [-30.0, -31.0],
                "longitude": [-70.0, -70.0],
                "depth": 10.0,
                "mag": [parent_mag, 4.5],
            }
        )

    small = nearest_neighbor(catalog(4.5), b=B, d=D, mc=MC, r_max_mode="gk", r_max_k=3)
    large = nearest_neighbor(catalog(7.0), b=B, d=D, mc=MC, r_max_mode="gk", r_max_k=3)

    assert pd.isna(small.set_index("id").loc["H", "parent_id"])
    assert large.set_index("id").loc["H", "parent_id"] == "P"


def test_el_nulo_usa_el_mismo_tope():
    catalog = random_catalog(n=200)
    free = randomize_catalog(catalog, b=B, d=D, mc=MC, n_repetitions=2, seed=3)
    capped = randomize_catalog(
        catalog, b=B, d=D, mc=MC, n_repetitions=2, seed=3, r_max_km=100
    )
    # Con tope hay eventos sin padre, o sea menos valores de eta que sin él.
    assert len(capped) < len(free)


def test_parametros_invalidos():
    catalog = random_catalog(n=20)
    with pytest.raises(ValueError):
        nearest_neighbor(catalog, b=B, d=D, mc=MC, r_max_km=-5)
    with pytest.raises(ValueError):
        nearest_neighbor(catalog, b=B, d=D, mc=MC, r_max_mode="otro")


def test_sufijo_de_nombre():
    assert method_suffix() == ""
    assert method_suffix(None, "fixed", 3.0) == ""
    assert method_suffix(300.0) == "_rmax=300"
    assert method_suffix(None, "gk", 3.0) == "_rmax=gk3"
    assert method_suffix(250, "gk", 2.5) == "_rmax=gk2.5-cap250"
