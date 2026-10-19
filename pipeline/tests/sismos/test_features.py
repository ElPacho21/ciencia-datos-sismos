"""Tests de las features previas al mainshock."""

import numpy as np
import pandas as pd
import pytest

from sismos.features import (
    PRIOR_RADIUS_KM,
    build_features,
    prior_seismicity,
    prior_suffix,
)

MC = 4.5
KM_PER_DEGREE = 111.195  # sobre la esfera de 6371 km
T0 = pd.Timestamp("2020-06-01", tz="UTC")


def event(event_id, days_before, km_north, mag=5.0, base=(-30.0, -70.0)):
    """Un evento a `km_north` km al norte de `base`, `days_before` días antes de T0."""
    return {
        "id": event_id,
        "time": T0 - pd.Timedelta(days=days_before),
        "latitude": base[0] + km_north / KM_PER_DEGREE,
        "longitude": base[1],
        "depth": 10.0,
        "mag": mag,
    }


def silver_for_mainshock():
    rows = [
        event("MAIN", 0, 0, mag=7.0),  # el propio sismo principal
        event("E1", 1, 10, mag=5.0),  # entra en las tres ventanas
        event("E2", 10, 50, mag=5.8),  # entra en 30 y 365
        event("E3", 200, 99, mag=4.6),  # entra sólo en 365
        event("E4", 5, 150, mag=6.0),  # fuera del radio de 100 km
        event("E5", 400, 10, mag=6.5),  # fuera de las tres ventanas
        event("E6", -3, 10, mag=5.0),  # POSTERIOR al sismo principal
        event("E7", 0.5, 10, mag=4.0),  # bajo Mc: no cuenta
    ]
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def summary_for(mainshocks):
    return pd.DataFrame(
        {
            "cluster_id": [m["id"] for m in mainshocks],
            "mainshock_id": [m["id"] for m in mainshocks],
            "mainshock_time": [m["time"] for m in mainshocks],
            "mainshock_lat": [m["latitude"] for m in mainshocks],
            "mainshock_lon": [m["longitude"] for m in mainshocks],
            "n_aftershocks": [3] * len(mainshocks),
        }
    )


def test_caso_armado_a_mano():
    silver = silver_for_mainshock()
    summary = summary_for([event("MAIN", 0, 0, mag=7.0)])

    # Ventana de 365 d dentro del catálogo: arranca 2 años antes de T0.
    result = prior_seismicity(summary, silver, MC, T0 - pd.Timedelta(days=800)).iloc[0]

    assert result["n_prior_2d"] == 1  # sólo E1
    assert result["n_prior_30d"] == 2  # E1, E2
    assert result["n_prior_365d"] == 3  # E1, E2, E3
    assert result["prior_max_mag_365d"] == 5.8  # E2; E5 (6.5) y E4 (6.0) no cuentan
    assert result["days_since_prior"] == pytest.approx(1.0)
    assert not result["prior_window_truncated"]


def test_ventana_truncada_y_nulos_estructurales():
    silver = silver_for_mainshock()
    lonely = event("LONELY", 0, 0, mag=6.0, base=(-10.0, -50.0))  # lejos de todo
    summary = summary_for([event("MAIN", 0, 0, mag=7.0), lonely])
    silver = pd.concat([silver, pd.DataFrame([lonely])]).sort_values("time")

    # El catálogo arranca 100 días antes de T0: los 365 días quedan truncados.
    result = prior_seismicity(summary, silver, MC, T0 - pd.Timedelta(days=100))

    assert result["prior_window_truncated"].tolist() == [True, True]
    isolated = result.iloc[1]
    assert isolated["n_prior_365d"] == 0
    assert np.isnan(isolated["prior_max_mag_365d"])
    assert np.isnan(isolated["days_since_prior"])


def test_el_radio_se_respeta_y_se_puede_cambiar():
    silver = silver_for_mainshock()
    summary = summary_for([event("MAIN", 0, 0, mag=7.0)])
    start = T0 - pd.Timedelta(days=800)

    narrow = prior_seismicity(summary, silver, MC, start, prior_radius_km=20).iloc[0]
    wide = prior_seismicity(summary, silver, MC, start, prior_radius_km=200).iloc[0]

    assert narrow["n_prior_365d"] == 1  # sólo E1 (10 km)
    assert wide["n_prior_365d"] == 4  # E1, E2, E3 y E4 (150 km)


def test_el_propio_mainshock_y_los_posteriores_no_cuentan():
    silver = silver_for_mainshock()
    summary = summary_for([event("MAIN", 0, 0, mag=7.0)])
    result = prior_seismicity(summary, silver, MC, T0 - pd.Timedelta(days=800)).iloc[0]
    # Si MAIN o E6 se colaran, n_prior_2d sería 2 o 3.
    assert result["n_prior_2d"] == 1


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def test_contra_fuerza_bruta_e_invariantes():
    rng = np.random.default_rng(7)
    n = 300
    start = pd.Timestamp("2015-01-01", tz="UTC")
    silver = pd.DataFrame(
        {
            "id": [f"e{i}" for i in range(n)],
            "time": start + pd.to_timedelta(np.sort(rng.uniform(0, 5 * 365 * 86400, n)), unit="s"),
            "latitude": rng.uniform(-31, -29, n),
            "longitude": rng.uniform(-71, -69, n),
            "depth": 10.0,
            "mag": 4.5 + rng.exponential(0.6, n),
        }
    )
    mains = silver.sample(40, random_state=1)
    summary = pd.DataFrame(
        {
            "cluster_id": mains["id"].to_numpy(),
            "mainshock_id": mains["id"].to_numpy(),
            "mainshock_time": mains["time"].to_numpy(),
            "mainshock_lat": mains["latitude"].to_numpy(),
            "mainshock_lon": mains["longitude"].to_numpy(),
            "n_aftershocks": 0,
        }
    )
    summary["mainshock_time"] = pd.to_datetime(summary["mainshock_time"], utc=True)

    result = prior_seismicity(summary, silver, MC, start, prior_radius_km=80)

    for i, row in summary.iterrows():
        d = haversine_km(row.mainshock_lat, row.mainshock_lon, silver.latitude, silver.longitude)
        age = (row.mainshock_time - silver["time"]).dt.total_seconds() / 86400
        inside = (d <= 80) & (age > 0) & (silver["id"] != row.mainshock_id)
        for window in (2, 30, 365):
            assert result.loc[i, f"n_prior_{window}d"] == int((inside & (age <= window)).sum())
        if inside.any():
            assert result.loc[i, "days_since_prior"] == pytest.approx(age[inside].min())
        else:
            assert np.isnan(result.loc[i, "days_since_prior"])

    assert (result["n_prior_2d"] <= result["n_prior_30d"]).all()
    assert (result["n_prior_30d"] <= result["n_prior_365d"]).all()


def test_build_features_agrega_columnas_al_final_sin_tocar_las_existentes():
    silver = silver_for_mainshock()
    summary = summary_for([event("MAIN", 0, 0, mag=7.0)])
    summary.loc[0, "n_aftershocks"] = 0

    out = build_features(summary, silver, MC, -5.8, T0 - pd.Timedelta(days=800))

    assert list(out.columns[: len(summary.columns)]) == list(summary.columns)
    assert out.loc[0, "region_mc"] == MC
    assert out.loc[0, "region_log10_eta0"] == -5.8
    assert not out.loc[0, "has_aftershock"]
    pd.testing.assert_frame_equal(out[summary.columns], summary)


def test_sufijo_del_radio():
    assert prior_suffix() == ""
    assert prior_suffix(PRIOR_RADIUS_KM) == ""
    assert prior_suffix(50) == "_prior=50"
    assert prior_suffix(75.5) == "_prior=75.5"
