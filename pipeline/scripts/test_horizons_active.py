"""Prueba de la Opción D: Tabla de horizontes (`horizons_combined`) condicionada
a secuencias con actividad observada hasta el instante t (`n_aft_t > 0`).

Evalúa la pregunta:
  "Dado que un sismo ya tuvo al menos 1 réplica en las primeras t horas
   (n_aft_t > 0), ¿seguirá teniendo réplicas hasta los 30 días (y_has_aft)
   y cuántas más faltan por ocurrir (y_aft)?"

Mide (con el mismo criterio y semáforo de `feature_baseline.py`):
1. Volumen y distribución del objetivo `y_aft` / `y_has_aft` por horizonte
   (t = 1h, 6h, 24h, 72h, 168h) y en la tabla larga combinada (`n_aft_t > 0`).
2. Semáforo de asociación (separación, eta², Spearman, Pearson, brecha) para
   las features estáticas del principal (`CORE + PRIOR`) y las dinámicas
   observadas hasta `t` (`n_aft_t`, `max_aft_mag_t`, `hours_since_last_t`,
   más las derivadas `mag_deficit_t = mainshock_mag - max_aft_mag_t` y
   `rate_per_hour_t = n_aft_t / horizon_hours`).
3. Baseline de regresión sobre `log1p(y_aft)` (contra mediana y media trivial)
   usando `KFold` en cada horizonte fijo y `GroupKFold(cluster_id)` en la
   tabla larga.
4. Baseline de clasificación sobre `y_has_aft` (¿habrá más réplicas después de t?)
   y conteo exacto de aciertos.
5. Importancia de variables por permutación.

Uso:
    python pipeline/scripts/test_horizons_active.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_baseline import CORE, CUTS, PRIOR, TARGET_BINS, TARGET_LABELS, _color, _eta_squared, _separation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLUSTERS = (
    ROOT
    / "pipeline/include/output/detector_replicas_api/delivery/clusters_combined_5regions-c4d8c04f_mainshock=largest_seed=1.csv"
)
DEFAULT_HORIZONS = (
    ROOT
    / "pipeline/include/output/detector_replicas_api/delivery/horizons_combined_5regions-c4d8c04f_mainshock=largest_seed=1.csv"
)

OBSERVED_T = (
    "n_aft_t",
    "max_aft_mag_t",
    "hours_since_last_t",
)

DERIVED_T = (
    "mag_deficit_t",   # Ley de Båth: mainshock_mag - max_aft_mag_t (~1.2 típico)
    "rate_per_hour_t", # Tasa media observada hasta t: n_aft_t / horizon_hours
    "silence_ratio_t", # Fracción del tiempo sin eventos: hours_since_last_t / horizon_hours
)


def load_active_horizons(clusters_path: Path, horizons_path: Path) -> pd.DataFrame:
    clusters = pd.read_csv(clusters_path)
    horizons = pd.read_csv(horizons_path)

    # Columnas de clusters que no están en horizons
    cluster_cols = ["cluster_id"] + [
        c for c in list(CORE) + list(PRIOR) + ["region_mc", "region_log10_eta0"]
        if c in clusters.columns and c not in horizons.columns
    ]
    merged = horizons.merge(clusters[cluster_cols], on="cluster_id", how="left")

    # Filtramos no censuradas en T y con actividad observada hasta t (n_aft_t > 0)
    active = merged[(~merged["censored_T"]) & (merged["n_aft_t"] > 0)].copy()

    # Features derivadas físicas (sin fuga: sólo usan lo ocurrido hasta t)
    active["mag_deficit_t"] = active["mainshock_mag"] - active["max_aft_mag_t"]
    active["rate_per_hour_t"] = active["n_aft_t"] / active["horizon_hours"]
    active["silence_ratio_t"] = active["hours_since_last_t"] / active["horizon_hours"]

    return active.reset_index(drop=True)


def design_matrix(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    parts = [df[list(columns)].astype(float)]
    if "region" in df.columns:
        parts.append(pd.get_dummies(df["region"], prefix="region", dtype=float))
    return pd.concat(parts, axis=1)


def print_overview(active: pd.DataFrame) -> None:
    print("=" * 92)
    print("1. PERFIL DE LA TABLA DE HORIZONTES CONDICIONADA A ACTIVIDAD TEMPRANA (n_aft_t > 0)")
    print("=" * 92)
    print(
        f"Total filas (todos los horizontes con n_aft_t > 0, sin censura): {len(active)} "
        f"| clusters únicos: {active['cluster_id'].nunique()}"
    )
    rows = []
    for h, g in active.groupby("horizon_hours"):
        y = g["y_aft"]
        rows.append(
            {
                "horizonte_h": int(h),
                "filas (n_aft_t>0)": len(g),
                "ceros_y_aft_%": round(100.0 * (y == 0).mean(), 1),
                "positivos_% (y_has_aft)": round(100.0 * (y > 0).mean(), 1),
                "media_y_aft": round(y.mean(), 2),
                "mediana_y_aft": round(y.median(), 1),
                "p75_y_aft": round(y.quantile(0.75), 1),
                "p90_y_aft": round(y.quantile(0.90), 1),
                "max_y_aft": int(y.max()),
            }
        )
    # Fila total combinada
    y_all = active["y_aft"]
    rows.append(
        {
            "horizonte_h": "TODOS (pool)",
            "filas (n_aft_t>0)": len(active),
            "ceros_y_aft_%": round(100.0 * (y_all == 0).mean(), 1),
            "positivos_% (y_has_aft)": round(100.0 * (y_all > 0).mean(), 1),
            "media_y_aft": round(y_all.mean(), 2),
            "mediana_y_aft": round(y_all.median(), 1),
            "p75_y_aft": round(y_all.quantile(0.75), 1),
            "p90_y_aft": round(y_all.quantile(0.90), 1),
            "max_y_aft": int(y_all.max()),
        }
    )
    print(pd.DataFrame(rows).set_index("horizonte_h").to_string())


def print_traffic_light(df: pd.DataFrame, label: str) -> None:
    print("\n" + "=" * 92)
    print(f"2. SEMÁFORO DE ASOCIACIÓN CONTRA `y_aft` — {label} (n = {len(df)})")
    print("=" * 92)
    target = df["y_aft"].astype(float)
    positive = df["y_has_aft"].astype(bool)
    groups = pd.cut(target, bins=TARGET_BINS, labels=TARGET_LABELS)

    candidate_cols = (
        list(OBSERVED_T)
        + list(DERIVED_T)
        + (["horizon_hours"] if df["horizon_hours"].nunique() > 1 else [])
        + list(CORE)
        + list(PRIOR)
    )

    rows = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        values = df[col].astype(float)
        pair = pd.DataFrame({"x": values, "y": target}).dropna()
        if len(pair) < 3 or not pair["x"].std():
            continue
        spearman = pair["x"].corr(pair["y"], method="spearman")
        pearson = pair["x"].corr(pair["y"])
        pearson_log = np.corrcoef(
            np.log1p(pair["x"].clip(lower=0)), np.log1p(pair["y"])
        )[0, 1]
        separation = _separation(values, positive)
        eta2 = _eta_squared(values, groups)
        gap = abs(spearman - pearson)
        rows.append(
            {
                "feature": col,
                "sep_est": f"{_color(separation, 'separacion')} {separation:6.3f}",
                "eta2": f"{_color(eta2, 'eta2')} {eta2:6.3f}",
                "spearman": f"{_color(spearman, 'correlacion')} {spearman:6.3f}",
                "pearson_log": f"{_color(pearson_log, 'correlacion')} {pearson_log:6.3f}",
                "pearson": f"{_color(pearson, 'correlacion')} {pearson:6.3f}",
                "brecha": f"{_color(gap, 'brecha')} {gap:6.3f}",
            }
        )
    print(pd.DataFrame(rows).set_index("feature").to_string())


def evaluate_regression_models(df: pd.DataFrame, is_pooled: bool = False) -> pd.DataFrame:
    y = df["y_aft"].astype(float).to_numpy()
    y_log = np.log1p(y)
    groups = df["cluster_id"].to_numpy() if is_pooled else None
    cv = GroupKFold(n_splits=5) if is_pooled else KFold(n_splits=5, shuffle=True, random_state=0)

    h_col = ("horizon_hours",) if is_pooled else ()
    feature_sets = [
        ("1. Sólo principal (CORE + región)", CORE + h_col),
        ("2. + sismicidad previa (CORE + PRIOR)", CORE + PRIOR + h_col),
        ("3. + observado hasta t (n_aft_t, max_mag_t, last_t)", CORE + PRIOR + OBSERVED_T + h_col),
        ("4. + derivadas hasta t (déficit Båth, tasa, silencio)", CORE + PRIOR + OBSERVED_T + DERIVED_T + h_col),
    ]

    med_log = np.median(y_log)
    mean_log = np.mean(y_log)

    rows = [
        {
            "modelo": f"trivial: mediana ({np.expm1(med_log):.0f} réplicas)",
            "R2_log": round(r2_score(y_log, np.full_like(y_log, med_log)), 3),
            "MAE_log": round(mean_absolute_error(y_log, np.full_like(y_log, med_log)), 3),
            "MAE_replicas": round(mean_absolute_error(y, np.full_like(y, np.median(y))), 2),
            "acierto_exacto_%": round(100.0 * (y.astype(int) == int(np.median(y))).mean(), 1),
            "acierto_+-1_rep_%": round(100.0 * (np.abs(y.astype(int) - int(np.median(y))) <= 1).mean(), 1),
        },
        {
            "modelo": "trivial: media de log1p",
            "R2_log": 0.000,
            "MAE_log": round(mean_absolute_error(y_log, np.full_like(y_log, mean_log)), 3),
            "MAE_replicas": round(mean_absolute_error(y, np.full_like(y, np.expm1(mean_log))), 2),
            "acierto_exacto_%": round(100.0 * (y.astype(int) == int(np.rint(np.expm1(mean_log)))).mean(), 1),
            "acierto_+-1_rep_%": round(100.0 * (np.abs(y.astype(int) - int(np.rint(np.expm1(mean_log)))) <= 1).mean(), 1),
        },
    ]

    for label, cols in feature_sets:
        X = design_matrix(df, cols)
        model = HistGradientBoostingRegressor(max_depth=6, random_state=0)
        pred_log = cross_val_predict(model, X, y_log, cv=cv, groups=groups)
        pred_rep = np.expm1(pred_log).clip(0)
        pred_int = np.rint(pred_rep).astype(int)
        y_int = y.astype(int)

        rows.append(
            {
                "modelo": label,
                "R2_log": round(r2_score(y_log, pred_log), 3),
                "MAE_log": round(mean_absolute_error(y_log, pred_log), 3),
                "MAE_replicas": round(mean_absolute_error(y, pred_rep), 2),
                "acierto_exacto_%": round(100.0 * (pred_int == y_int).mean(), 1),
                "acierto_+-1_rep_%": round(100.0 * (np.abs(pred_int - y_int) <= 1).mean(), 1),
            }
        )

    # Agregamos versión entrenada con loss='absolute_error' sobre el set completo de features
    X_full = design_matrix(df, CORE + PRIOR + OBSERVED_T + DERIVED_T + h_col)
    model_mae = HistGradientBoostingRegressor(loss="absolute_error", max_depth=6, random_state=0)
    pred_log_mae = cross_val_predict(model_mae, X_full, y_log, cv=cv, groups=groups)
    pred_rep_mae = np.expm1(pred_log_mae).clip(0)
    pred_int_mae = np.rint(pred_rep_mae).astype(int)
    rows.append(
        {
            "modelo": "5. Completo con loss='absolute_error' (optimiza MAE)",
            "R2_log": round(r2_score(y_log, pred_log_mae), 3),
            "MAE_log": round(mean_absolute_error(y_log, pred_log_mae), 3),
            "MAE_replicas": round(mean_absolute_error(y, pred_rep_mae), 2),
            "acierto_exacto_%": round(100.0 * (pred_int_mae == y.astype(int)).mean(), 1),
            "acierto_+-1_rep_%": round(100.0 * (np.abs(pred_int_mae - y.astype(int)) <= 1).mean(), 1),
        }
    )

    return pd.DataFrame(rows).set_index("modelo")


def evaluate_classification_models(df: pd.DataFrame, is_pooled: bool = False) -> pd.DataFrame:
    """Evalúa clasificación binaria sobre `y_has_aft` (¿habrá al menos 1 réplica más entre t y T?)."""
    y = df["y_has_aft"].astype(int).to_numpy()
    groups = df["cluster_id"].to_numpy() if is_pooled else None
    cv = GroupKFold(n_splits=5) if is_pooled else StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    h_col = ("horizon_hours",) if is_pooled else ()

    feature_sets = [
        ("1. Sólo principal (CORE + región)", CORE + h_col),
        ("2. + sismicidad previa (CORE + PRIOR)", CORE + PRIOR + h_col),
        ("3. + observado hasta t (OBSERVED_T)", CORE + PRIOR + OBSERVED_T + h_col),
        ("4. + derivadas hasta t (OBSERVED_T + DERIVED_T)", CORE + PRIOR + OBSERVED_T + DERIVED_T + h_col),
    ]

    trivial_acc = max(y.mean(), 1.0 - y.mean())
    rows = [
        {
            "modelo_clasificacion": f"trivial: clase mayoritaria ({'False/0' if y.mean() < 0.5 else 'True/1'})",
            "ROC_AUC": 0.500,
            "PR_AUC": round(float(y.mean()), 3),
            "Acc@0.5": round(float(trivial_acc), 3),
            "aciertos_totales": f"{int(round(trivial_acc * len(y)))}/{len(y)}",
            "F1@0.5": round(f1_score(y, np.ones_like(y) if y.mean() >= 0.5 else np.zeros_like(y)), 3),
            "Bal_Acc": 0.500,
            "TP (detectados)": f"{int(y.sum()) if y.mean()>=0.5 else 0}/{int(y.sum())}",
            "TN (ceros acertados)": f"{0 if y.mean()>=0.5 else int((1-y).sum())}/{int((1-y).sum())}",
        }
    ]

    for label, cols in feature_sets:
        X = design_matrix(df, cols)
        clf = HistGradientBoostingClassifier(max_depth=6, random_state=0)
        prob = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        rows.append(
            {
                "modelo_clasificacion": label,
                "ROC_AUC": round(roc_auc_score(y, prob), 3),
                "PR_AUC": round(average_precision_score(y, prob), 3),
                "Acc@0.5": round(accuracy_score(y, pred), 3),
                "aciertos_totales": f"{tn + tp}/{len(y)} ({100*(tn+tp)/len(y):.1f}%)",
                "F1@0.5": round(f1_score(y, pred), 3),
                "Bal_Acc": round(balanced_accuracy_score(y, pred), 3),
                "TP (detectados)": f"{tp}/{int(y.sum())} ({100*tp/y.sum():.1f}%)",
                "TN (ceros acertados)": f"{tn}/{int((1-y).sum())} ({100*tn/(1-y).sum():.1f}%)",
            }
        )
    return pd.DataFrame(rows).set_index("modelo_clasificacion")


def print_importance(df: pd.DataFrame, is_pooled: bool = False) -> None:
    y_log = np.log1p(df["y_aft"].astype(float).to_numpy())
    h_col = ("horizon_hours",) if is_pooled else ()
    cols = CORE + PRIOR + OBSERVED_T + DERIVED_T + h_col
    X = design_matrix(df, cols)
    model = HistGradientBoostingRegressor(max_depth=6, random_state=0).fit(X, y_log)
    result = permutation_importance(
        model, X, y_log, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error"
    )
    order = np.argsort(result.importances_mean)[::-1]
    print("\nImportancia por permutación (cuánto empeora el MAE_log al desordenar la columna):")
    for i in order[:12]:
        print(f"  {X.columns[i]:<26} {result.importances_mean[i]:.4f}")


def main() -> int:
    active = load_active_horizons(DEFAULT_CLUSTERS, DEFAULT_HORIZONS)
    print_overview(active)

    # Semáforo en t=1h y en tabla combinada (todos los horizontes con n_aft_t > 0)
    at_1h = active[active["horizon_hours"] == 1].reset_index(drop=True)
    at_6h = active[active["horizon_hours"] == 6].reset_index(drop=True)
    print_traffic_light(at_1h, "Horizonte fijo t = 1 h (n_aft_t > 0)")
    print_traffic_light(active, "Tabla larga combinada (todos los horizontes con n_aft_t > 0)")

    print("\n" + "=" * 92)
    print(f"3A. REGRESIÓN `log1p(y_aft)` EN HORIZONTE t = 1 h (n = {len(at_1h)}, 46.8% tienen más réplicas)")
    print("=" * 92)
    print(evaluate_regression_models(at_1h, is_pooled=False).to_string())

    print("\n" + "=" * 92)
    print(f"3B. REGRESIÓN `log1p(y_aft)` EN HORIZONTE t = 6 h (n = {len(at_6h)}, >1000 filas!)")
    print("=" * 92)
    print(evaluate_regression_models(at_6h, is_pooled=False).to_string())

    print("\n" + "=" * 92)
    print(f"3C. REGRESIÓN `log1p(y_aft)` EN TABLA LARGA COMBINADA (n = {len(active)}, GroupKFold por cluster_id)")
    print("=" * 92)
    print(evaluate_regression_models(active, is_pooled=True).to_string())
    print_importance(active, is_pooled=True)

    print("\n" + "=" * 92)
    print(f"4A. CLASIFICACIÓN BINARIA `y_has_aft` EN HORIZONTE t = 1 h (n = {len(at_1h)}, balance 47% / 53%!)")
    print("=" * 92)
    print(evaluate_classification_models(at_1h, is_pooled=False).to_string())

    print("\n" + "=" * 92)
    print(f"4B. CLASIFICACIÓN BINARIA `y_has_aft` EN HORIZONTE t = 6 h (n = {len(at_6h)}, >1000 filas)")
    print("=" * 92)
    print(evaluate_classification_models(at_6h, is_pooled=False).to_string())

    print("\n" + "=" * 92)
    print(f"4C. CLASIFICACIÓN BINARIA `y_has_aft` EN TABLA LARGA COMBINADA (n = {len(active)}, GroupKFold)")
    print("=" * 92)
    print(evaluate_classification_models(active, is_pooled=True).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
