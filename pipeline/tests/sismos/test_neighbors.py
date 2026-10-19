"""Tests del vecino más cercano sobre catálogos sintéticos con respuesta conocida."""

import numpy as np
import pandas as pd
import pytest

from sismos.neighbors import nearest_neighbor

B, D, MC = 1.0, 1.5, 4.0


def make_catalog(rows):
    """rows: (id, horas desde el inicio, lat, lon, mag)."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
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


def test_replica_cercana_elige_al_sismo_grande_como_padre():
    # A es grande; B y C son chicos y lejos entre sí; D cae pegado a A.
    catalog = make_catalog(
        [
            ("A", 0, -30.0, -70.0, 7.0),
            ("B", 10, -40.0, -65.0, 4.5),
            ("C", 20, -22.0, -60.0, 4.5),
            ("D", 30, -30.05, -70.05, 4.5),
        ]
    )
    paired = nearest_neighbor(catalog, b=B, d=D, mc=MC).set_index("id")

    assert pd.isna(paired.loc["A", "parent_id"])
    assert paired.loc["D", "parent_id"] == "A"
    assert paired.loc["D", "parent_mag"] == 7.0


def test_eta_es_el_producto_de_T_y_R():
    catalog = make_catalog(
        [("A", 0, -30.0, -70.0, 6.0), ("B", 5, -30.2, -70.1, 4.5)]
    )
    paired = nearest_neighbor(catalog, b=B, d=D, mc=MC).set_index("id")

    assert paired.loc["B", "log10_eta"] == pytest.approx(
        paired.loc["B", "log10_T"] + paired.loc["B", "log10_R"]
    )


def test_el_primer_evento_no_tiene_padre():
    catalog = make_catalog(
        [("A", 0, -30.0, -70.0, 5.0), ("B", 1, -31.0, -71.0, 5.0)]
    )
    paired = nearest_neighbor(catalog, b=B, d=D, mc=MC)

    assert paired["parent_id"].isna().tolist() == [True, False]
    assert np.isnan(paired["eta"].iloc[0])
