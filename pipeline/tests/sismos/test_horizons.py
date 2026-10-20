"""Objetivos por horizonte: caso armado a mano con respuestas escritas e invariantes."""

import numpy as np
import pandas as pd
import pytest

from sismos.horizons import (
    COLUMNS,
    build_horizons,
    check_invariants,
    horizon_suffix,
)

T0 = pd.Timestamp("2020-01-01T00:00:00", tz="UTC")
END = pd.Timestamp("2021-01-01T00:00:00", tz="UTC")


def _events(cluster_id, main_id, rows):
    """rows: (id, horas respecto del principal, magnitud)."""
    return pd.DataFrame(
        [
            {
                "cluster_id": cluster_id,
                "id": i,
                "time": T0 + pd.Timedelta(hours=h),
                "mag": m,
            }
            for i, h, m in rows
        ]
    )


@pytest.fixture
def sintetico():
    # Réplicas a 0,5 h, 10 h, 3 d (72 h) y 40 d; más un premonitor a -2 h.
    a = _events(
        "A",
        "A_main",
        [
            ("A_fore", -2, 4.6),
            ("A_main", 0, 7.0),
            ("A_1", 0.5, 5.0),
            ("A_2", 10, 4.8),
            ("A_3", 72, 4.9),
            ("A_4", 40 * 24, 5.5),
        ],
    )
    # Un cluster sin réplicas.
    b = _events("B", "B_main", [("B_main", 0, 6.0)])
    events = pd.concat([a, b], ignore_index=True)
    summary = pd.DataFrame(
        {
            "cluster_id": ["A", "B"],
            "mainshock_id": ["A_main", "B_main"],
            "mainshock_time": [T0, T0],
            "n_aftershocks": [4, 0],
            "duration_days": [40.1, 0.0],
        }
    )
    return summary, events


def test_respuestas_escritas_a_mano(sintetico):
    summary, events = sintetico
    out = build_horizons(summary, events, END)
    a = out[out["cluster_id"] == "A"].set_index("horizon_hours")

    assert list(a["n_aft_t"]) == [1, 1, 2, 3, 3]
    assert list(a["y_aft"]) == [2, 2, 1, 0, 0]
    assert list(a["n_aft_T"]) == [3] * 5
    # Horas desde el último evento visto (el principal cuenta si no hubo réplicas).
    assert list(a["hours_since_last_t"]) == [0.5, 5.5, 14.0, 0.0, 96.0]
    assert a.loc[1, "max_aft_mag_t"] == 5.0
    assert a.loc[24, "max_aft_mag_t"] == 5.0
    assert not a["censored_T"].any()


def test_cluster_sin_replicas(sintetico):
    summary, events = sintetico
    b = build_horizons(summary, events, END).query("cluster_id == 'B'")
    assert (b["n_aft_t"] == 0).all() and (b["y_aft"] == 0).all()
    assert b["max_aft_mag_t"].isna().all()
    # Sin réplicas el último evento es el principal: pasaron exactamente t horas.
    assert list(b["hours_since_last_t"]) == list(b["horizon_hours"])


def test_el_premonitor_no_cuenta_como_replica(sintetico):
    summary, events = sintetico
    out = build_horizons(summary, events, END)
    assert out[(out["cluster_id"] == "A") & (out["horizon_hours"] == 168)]["n_aft_t"].item() == 3


def test_censura_se_marca_y_no_se_borra(sintetico):
    summary, events = sintetico
    # El catálogo termina 10 días después del principal: T = 30 d no se ve completo.
    end = T0 + pd.Timedelta(days=10)
    out = build_horizons(summary, events, end)
    assert len(out) == 10
    assert out["censored_T"].all()
    # La feature de 7 d sí se ve entera; la de 24 h también.
    assert not out["censored_t"].any()
    short = T0 + pd.Timedelta(hours=5)
    assert build_horizons(summary, events, short).query("horizon_hours > 5")["censored_t"].all()


def test_invariantes_y_columnas(sintetico):
    summary, events = sintetico
    out = build_horizons(summary, events, END)
    assert list(out.columns) == list(COLUMNS)
    check_invariants(out, summary)

    # La secuencia B cabe en T: n_aft_T == n_aftershocks. La A no (hay una a 40 d).
    broken = out.copy()
    broken.loc[broken["cluster_id"] == "B", "n_aft_T"] = 1
    with pytest.raises(ValueError):
        check_invariants(broken, summary)


def test_horizonte_fuera_de_rango_falla(sintetico):
    summary, events = sintetico
    with pytest.raises(ValueError):
        build_horizons(summary, events, END, horizons_hours=(1, 24 * 40))


def test_sufijo_de_T():
    assert horizon_suffix(30) == ""
    assert horizon_suffix(14) == "_T=14"


def test_datos_reales_de_referencia():
    """Sobre la corrida dorada: invariantes y n_aft_T == n_aftershocks donde corresponde."""
    import glob
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    ev_files = glob.glob(str(root / "events_*seed=1.csv"))
    sm_files = glob.glob(str(root / "clusters_*seed=1.csv"))
    if not ev_files or not sm_files:
        pytest.skip("reference csv files are not in the repo root")

    ev = pd.read_csv(ev_files[0])
    ev["time"] = pd.to_datetime(ev["time"], format="ISO8601", utc=True)
    sm = pd.read_csv(sm_files[0])
    sm["mainshock_time"] = pd.to_datetime(sm["mainshock_time"], format="ISO8601", utc=True)

    out = build_horizons(sm, ev, "2026-01-01")
    check_invariants(out, sm)
    assert len(out) == 5 * len(sm)
    assert out["cluster_id"].nunique() == len(sm)
    assert np.isfinite(out["hours_since_last_t"]).all()
