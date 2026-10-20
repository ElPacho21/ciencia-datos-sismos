"""Tests de regiones: borde, validación y concatenación."""

import numpy as np
import pandas as pd
import pytest

from sismos.features import build_features
from sismos.regions import (
    BORDER_BUFFER_KM,
    add_border_columns,
    border_suffix,
    combine_regions,
    distance_to_border_km,
    validate_regions,
)

R = 6371.0
KM_PER_DEGREE = np.radians(1.0) * R


def region(name, lat0, lat1, lon0, lon1):
    return {
        "name": name,
        "minlatitude": lat0,
        "maxlatitude": lat1,
        "minlongitude": lon0,
        "maxlongitude": lon1,
        "starttime": "2000-01-01",
        "endtime": "2020-01-01",
        "minmagnitude": 4.0,
    }


def test_distancia_a_un_borde_de_latitud():
    # 2 grados al norte del borde sur, lejos de los meridianos.
    d = distance_to_border_km(-30.0, -65.0, (-32.0, -10.0, -90.0, -40.0))
    assert d == pytest.approx(2 * KM_PER_DEGREE)


def test_distancia_a_un_meridiano_se_achica_con_la_latitud():
    # 1 grado de longitud del borde: en el ecuador ~111 km, a 60 grados ~ la mitad.
    equator = distance_to_border_km(0.0, 1.0, (-5, 5, 0.0, 30.0))
    high = distance_to_border_km(60.0, 1.0, (55, 65, 0.0, 30.0))
    assert equator == pytest.approx(KM_PER_DEGREE, rel=1e-3)
    assert high == pytest.approx(KM_PER_DEGREE * 0.5, rel=1e-2)


def test_el_borde_mas_cercano_es_el_minimo():
    d = distance_to_border_km(0.0, 10.0, (-1.0, 40.0, -50.0, 50.0))
    assert d == pytest.approx(1 * KM_PER_DEGREE)  # el borde sur, a 1 grado


def test_near_border_y_columnas_al_final():
    summary = pd.DataFrame(
        {"mainshock_lat": [-31.5, -20.0], "mainshock_lon": [-65.0, -65.0], "x": [1, 2]}
    )
    out = add_border_columns(summary, (-32.0, -10.0, -90.0, -40.0), buffer_km=100)
    assert list(out.columns[:3]) == list(summary.columns)
    assert out["near_border"].tolist() == [True, False]  # 55 km vs 1.100 km


def test_build_features_con_bbox_agrega_el_borde():
    start = pd.Timestamp("2020-01-01", tz="UTC")
    silver = pd.DataFrame(
        {
            "id": ["M"],
            "time": [start],
            "latitude": [-31.5],
            "longitude": [-65.0],
            "depth": 10.0,
            "mag": [6.0],
        }
    )
    summary = pd.DataFrame(
        {
            "cluster_id": ["M"],
            "mainshock_id": ["M"],
            "mainshock_time": [start],
            "mainshock_lat": [-31.5],
            "mainshock_lon": [-65.0],
            "n_aftershocks": [0],
        }
    )
    with_box = build_features(
        summary, silver, 4.5, -5.8, "2015-01-01", bbox=(-32, -10, -90, -40)
    )
    without = build_features(summary, silver, 4.5, -5.8, "2015-01-01")

    assert {"dist_to_border_km", "near_border"} <= set(with_box.columns)
    assert not {"dist_to_border_km", "near_border"} & set(without.columns)
    assert bool(with_box.loc[0, "near_border"]) is True


def test_validacion_de_regiones():
    # Rectángulos que sólo se tocan en el borde no se solapan.
    validate_regions([region("a", 0, 10, 0, 10), region("b", 0, 10, 10, 20)])

    with pytest.raises(ValueError, match="overlap"):
        validate_regions([region("a", 0, 10, 0, 10), region("b", 5, 15, 5, 15)])
    with pytest.raises(ValueError, match="unique"):
        validate_regions([region("a", 0, 10, 0, 10), region("a", 20, 30, 0, 10)])
    with pytest.raises(ValueError, match="antimeridian"):
        validate_regions([region("a", 0, 10, 170, -170)])
    with pytest.raises(ValueError, match="missing"):
        validate_regions([{"name": "a"}])


def frame(ids, mainshock_ids=None):
    return pd.DataFrame(
        {"cluster_id": ids, "mainshock_id": mainshock_ids or ids, "n_aftershocks": 0}
    )


def test_combine_agrega_region_y_conserva_filas():
    out = combine_regions({"a": frame(["x1", "x2"]), "b": frame(["y1"])})
    assert out["region"].tolist() == ["a", "a", "b"]
    assert len(out) == 3
    assert out.columns[-1] == "region"


def test_combine_falla_si_un_cluster_o_mainshock_se_repite():
    with pytest.raises(ValueError, match="cluster_id"):
        combine_regions({"a": frame(["x"]), "b": frame(["x"])})
    with pytest.raises(ValueError, match="mainshock_id"):
        combine_regions({"a": frame(["x1"], ["m"]), "b": frame(["y1"], ["m"])})


def test_combine_falla_si_las_columnas_difieren():
    other = frame(["y"]).assign(extra=1)
    with pytest.raises(ValueError, match="different columns"):
        combine_regions({"a": frame(["x"]), "b": other})


def test_sufijo_del_borde():
    assert border_suffix() == ""
    assert border_suffix(BORDER_BUFFER_KM) == ""
    assert border_suffix(150) == "_border=150"


def test_el_yaml_versionado_es_valido():
    """El `regions.yaml` real pasa la validación: sin solapes ni antimeridiano."""
    from sismos.regions import load_regions, regions_tag

    regions = load_regions()
    assert len(regions) == 5
    assert regions_tag(regions).startswith("5regions-")
