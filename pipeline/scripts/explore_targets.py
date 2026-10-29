"""Explora y compara qué otras columnas (o formulaciones) se pueden predecir
sobre clusters_combined_5regions para evitar el problema del 89.4% de ceros
en `n_aftershocks`.

Reutiliza la lógica y el diseño de `feature_baseline.py` (mismos bloques CORE y
PRIOR, mismo HistGradientBoosting y validación cruzada 5-fold) y evalúa:

1. Perfil de ceros y nulos de todas las columnas de `clusters_combined_*.csv`.
2. Todas las columnas candidatas a objetivo de regresión en las 19.570 filas
   (`n_aftershocks`, `duration_days`, `extent_km`, `max_generation`, `n_events`,
   `mainshock_depth`, `mainshock_mag`), comparando el MAE del modelo contra la
   mediana trivial y la media trivial.
3. Objetivos de clasificación binaria (`has_aftershock`, `root_is_mainshock`)
   con ROC-AUC, PR-AUC, F1 y ganancia sobre el clasificador trivial.
4. Enfoque en 2 etapas (Hurdle) / subconjuntos sin inflación de ceros:
   - Etapa 2: predecir `n_aftershocks`, `duration_days` o `extent_km` sólo en
     secuencias activas (`has_aftershock == True`, 2.069 filas, 0% ceros).
   - Filtro causal por magnitud (`mainshock_mag >= 5.0` con 4.764 filas, o
     `mainshock_mag >= 5.5` con 1.454 filas donde el 57.2% tiene réplicas).
   - Predicción en dos etapas combinada sobre las 19.570 filas (clasificador
     `has_aftershock` + regresor en positivas).
5. Semáforo de asociación (estilo `feature_baseline.py`) para los mejores
   objetivos alternativos.

Uso:
    python pipeline/scripts/explore_targets.py \\
        pipeline/include/output/detector_replicas_api/delivery/clusters_combined_5regions-c4d8c04f_mainshock=largest_seed=1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

# Importamos constantes y helpers de feature_baseline.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_baseline import CORE, CUTS, DERIVED, PRIOR, _color, _eta_squared, _separation, design


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de tipo, nulos, ceros y percentiles de cada columna."""
    rows = []
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        if pd.api.types.is_bool_dtype(s):
            true_pct = 100.0 * s.mean()
            rows.append(
                {
                    "columna": col,
                    "tipo": "bool",
                    "nulos_%": round(100.0 * n_null / len(df), 1),
                    "ceros_o_False_%": round(100.0 - true_pct, 1),
                    "media_o_True%": f"{true_pct:.1f}% True",
                    "mediana": str(bool(s.mode().iloc[0])),
                    "p90": "-",
                    "max": "-",
                }
            )
        elif pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            zero_pct = 100.0 * (valid == 0).mean() if len(valid) else 0.0
            rows.append(
                {
                    "columna": col,
                    "tipo": str(s.dtype),
                    "nulos_%": round(100.0 * n_null / len(df), 1),
                    "ceros_o_False_%": round(zero_pct, 1),
                    "media_o_True%": f"{valid.mean():.2f}",
                    "mediana": f"{valid.median():.2f}",
                    "p90": f"{valid.quantile(0.9):.2f}",
                    "max": f"{valid.max():.2f}",
                }
            )
        else:
            rows.append(
                {
                    "columna": col,
                    "tipo": "str/cat",
                    "nulos_%": round(100.0 * n_null / len(df), 1),
                    "ceros_o_False_%": "-",
                    "media_o_True%": f"{s.nunique()} únicos",
                    "mediana": "-",
                    "p90": "-",
                    "max": "-",
                }
            )
    return pd.DataFrame(rows).set_index("columna")


def eval_regression_candidates(df: pd.DataFrame, label_prefix: str = "") -> pd.DataFrame:
    """Evalúa candidatos de regresión comparando contra mediana y media trivial."""
    candidates = [
        ("n_aftershocks", True, CORE + PRIOR),
        ("duration_days", True, CORE + PRIOR),
        ("extent_km", True, CORE + PRIOR),
        ("max_generation", True, CORE + PRIOR),
        ("n_events", True, CORE + PRIOR),
        (
            "mainshock_depth",
            True,
            ("mainshock_mag", "mainshock_lat", "mainshock_lon", "dist_to_border_km") + PRIOR,
        ),
        (
            "mainshock_mag",
            False,
            ("mainshock_depth", "mainshock_lat", "mainshock_lon", "dist_to_border_km") + PRIOR,
        ),
    ]
    rows = []
    for target_col, use_log, feat_cols in candidates:
        if target_col not in df.columns:
            continue
        sub = df.dropna(subset=[target_col]).reset_index(drop=True)
        y_raw = sub[target_col].astype(float).to_numpy()
        y = np.log1p(np.clip(y_raw, 0, None)) if use_log else y_raw
        X = design(sub, feat_cols)

        model = HistGradientBoostingRegressor(max_depth=6, random_state=0)
        pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))

        med_val = np.median(y)
        mean_val = np.mean(y)
        mae_model = mean_absolute_error(y, pred)
        mae_med = mean_absolute_error(y, np.full_like(y, med_val))
        mae_mean = mean_absolute_error(y, np.full_like(y, mean_val))

        pred_orig = np.expm1(pred).clip(0) if use_log else pred
        mae_orig_model = mean_absolute_error(y_raw, pred_orig)
        mae_orig_med = mean_absolute_error(y_raw, np.full_like(y_raw, np.median(y_raw)))

        rows.append(
            {
                "objetivo": f"{label_prefix}{target_col}" + (" (log1p)" if use_log else ""),
                "filas": len(sub),
                "ceros_%": round(100.0 * (y_raw == 0).mean(), 1),
                "R2": round(r2_score(y, pred), 3),
                "MAE_modelo": round(mae_model, 3),
                "MAE_trivial_med": round(mae_med, 3),
                "MAE_trivial_media": round(mae_mean, 3),
                "gana_a_mediana?": "SI" if mae_model < mae_med - 1e-4 else "NO (empata/pierde)",
                "MAE_escala_orig": f"{mae_orig_model:.2f} (med={mae_orig_med:.2f})",
            }
        )
    return pd.DataFrame(rows).set_index("objetivo")


def eval_classification_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Evalúa objetivos binarios en toda la tabla y por corte de magnitud."""
    configs = [
        ("has_aftershock (todas las filas)", df, "has_aftershock", CORE + PRIOR),
        ("has_aftershock (M >= 5.0)", df[df["mainshock_mag"] >= 5.0], "has_aftershock", CORE + PRIOR),
        ("has_aftershock (M >= 5.5)", df[df["mainshock_mag"] >= 5.5], "has_aftershock", CORE + PRIOR),
        ("root_is_mainshock (todas)", df, "root_is_mainshock", CORE + PRIOR),
    ]
    rows = []
    for name, sub_df, target_col, feat_cols in configs:
        sub = sub_df.dropna(subset=[target_col]).reset_index(drop=True)
        y = sub[target_col].astype(int).to_numpy()
        X = design(sub, feat_cols)

        clf = HistGradientBoostingClassifier(max_depth=6, random_state=0)
        prob = cross_val_predict(
            clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba"
        )[:, 1]
        pred_05 = (prob >= 0.5).astype(int)

        best_f1, best_th = 0.0, 0.5
        for th in np.linspace(0.10, 0.90, 81):
            f1 = f1_score(y, (prob >= th).astype(int))
            if f1 > best_f1:
                best_f1, best_th = f1, float(th)

        pred_best = (prob >= best_th).astype(int)
        trivial_acc = max(y.mean(), 1.0 - y.mean())

        rows.append(
            {
                "objetivo_binario": name,
                "filas": len(sub),
                "positivos_%": round(100.0 * y.mean(), 1),
                "ROC_AUC": round(roc_auc_score(y, prob), 3),
                "PR_AUC": round(average_precision_score(y, prob), 3),
                "Acc@0.5": round(accuracy_score(y, pred_05), 3),
                "Acc_trivial": round(trivial_acc, 3),
                "F1@0.5": round(f1_score(y, pred_05), 3),
                "mejor_F1": f"{best_f1:.3f} (th={best_th:.2f})",
                "Bal_Acc": round(balanced_accuracy_score(y, pred_best), 3),
            }
        )
    return pd.DataFrame(rows).set_index("objetivo_binario")


def eval_hurdle_two_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Evalúa un modelo en 2 etapas (Hurdle) sobre las 19.570 filas para n_aftershocks."""
    y = df["n_aftershocks"].astype(float).to_numpy()
    y_log = np.log1p(y)
    y_bin = (y > 0).astype(int)
    X = design(df, CORE + PRIOR)

    kf = KFold(5, shuffle=True, random_state=0)
    prob_pos = np.zeros(len(df))
    pred_pos_log = np.zeros(len(df))
    pred_std_log = np.zeros(len(df))

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_log, y_tr_bin = y_log[train_idx], y_bin[train_idx]

        # Modelo estándar (1 sola etapa)
        reg_all = HistGradientBoostingRegressor(max_depth=6, random_state=0)
        reg_all.fit(X_tr, y_tr_log)
        pred_std_log[test_idx] = reg_all.predict(X_te)

        # Etapa 1: Clasificador P(has_aftershock)
        clf = HistGradientBoostingClassifier(max_depth=6, random_state=0)
        clf.fit(X_tr, y_tr_bin)
        prob_pos[test_idx] = clf.predict_proba(X_te)[:, 1]

        # Etapa 2: Regresor entrenado SOLO donde hay réplicas (y > 0)
        pos_mask = y_tr_bin == 1
        reg_pos = HistGradientBoostingRegressor(loss="absolute_error", max_depth=6, random_state=0)
        reg_pos.fit(X_tr.iloc[pos_mask], y_tr_log[pos_mask])
        pred_pos_log[test_idx] = reg_pos.predict(X_te)

    rows = [
        {
            "estrategia": "trivial: decir siempre 0 réplicas",
            "R2_log": round(r2_score(y_log, np.zeros_like(y_log)), 3),
            "MAE_log": round(mean_absolute_error(y_log, np.zeros_like(y_log)), 4),
            "MAE_replicas": round(mean_absolute_error(y, np.zeros_like(y)), 3),
        },
        {
            "estrategia": "1 etapa: regresor directo (squared_error)",
            "R2_log": round(r2_score(y_log, pred_std_log), 3),
            "MAE_log": round(mean_absolute_error(y_log, pred_std_log), 4),
            "MAE_replicas": round(mean_absolute_error(y, np.expm1(pred_std_log).clip(0)), 3),
        },
    ]

    # En 2 etapas: si P(has_aftershock) < umbral -> 0, si >= umbral -> pred_pos_log
    for th in [0.35, 0.45, 0.50]:
        gated_log = np.where(prob_pos >= th, pred_pos_log, 0.0)
        rows.append(
            {
                "estrategia": f"2 etapas (hurdle th={th:.2f}): 0 si P<{th:.2f}, sino regresor activo",
                "R2_log": round(r2_score(y_log, gated_log), 3),
                "MAE_log": round(mean_absolute_error(y_log, gated_log), 4),
                "MAE_replicas": round(mean_absolute_error(y, np.expm1(gated_log).clip(0)), 3),
            }
        )
    return pd.DataFrame(rows).set_index("estrategia")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "clusters_csv",
        nargs="?",
        default="pipeline/include/output/detector_replicas_api/delivery/clusters_combined_5regions-c4d8c04f_mainshock=largest_seed=1.csv",
        help="Ruta al CSV de clusters combinados",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.clusters_csv)
    print("=" * 88)
    print(f"1. PERFIL DE COLUMNAS EN {Path(args.clusters_csv).name} ({len(df)} filas)")
    print("=" * 88)
    print(profile_columns(df).to_string())

    print("\n" + "=" * 88)
    print("2. REGRESIÓN SOBRE TODAS LAS FILAS (n = 19.570): ¿QUÉ COLUMNA LE GANA AL TRIVIAL?")
    print("=" * 88)
    print(eval_regression_candidates(df).to_string())

    print("\n" + "=" * 88)
    print("3. CLASIFICACIÓN BINARIA: PREDECIR `has_aftershock` EN VEZ DEL CONTEO")
    print("=" * 88)
    print(eval_classification_candidates(df).to_string())

    print("\n" + "=" * 88)
    print("4. REGRESIÓN CONDICIONADA (ETAPA 2: SÓLO SECUENCIAS CON RÉPLICAS, n = 2.069, 0% CEROS)")
    print("=" * 88)
    df_act = df[df["has_aftershock"]].reset_index(drop=True)
    print(eval_regression_candidates(df_act, label_prefix="[Activas] ").to_string())

    print("\n" + "=" * 88)
    print("5. REGRESIÓN EN SISMOS MODERADOS/FUERTES (mainshock_mag >= 5.5, n = 1.454)")
    print("=" * 88)
    df_m55 = df[df["mainshock_mag"] >= 5.5].reset_index(drop=True)
    print(eval_regression_candidates(df_m55, label_prefix="[M>=5.5] ").to_string())

    print("\n" + "=" * 88)
    print("6. MODELO EN 2 ETAPAS (HURDLE) SOBRE LAS 19.570 FILAS")
    print("=" * 88)
    print(eval_hurdle_two_stage(df).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
