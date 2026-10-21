"""¿Aportan estas features? Números para la defensa, no opiniones.

Sobre el csv de secuencias publicado por el DAG, mide qué agrega el bloque de
sismicidad previa (`n_prior_*`, `days_since_prior`, ...) al objetivo
`n_aftershocks`:

- el resumen del objetivo (es 89 % ceros, y eso condiciona cualquier métrica);
- la correlación de cada feature, global y dentro de cada región, con Pearson
  sobre `log1p` además del Spearman — un Pearson alto con Spearman bajo son
  cuatro eventos extremos tirando de la recta;
- la tabla magnitud x sismicidad previa de 2 días, que es donde se ve que la
  feature no está confundida con la magnitud;
- el baseline del modelo: el mismo HistGradientBoosting sin y con las features
  nuevas, más dos predicciones triviales contra las que hay que compararse;
- el semáforo de asociación: cada feature en verde, amarillo o rojo según los
  umbrales de siempre, para el cribado rápido y para la defensa.

El objetivo se modela como `log1p(n_aftershocks)`: en conteos crudos el error
absoluto lo dominan las secuencias gigantes (Japón 2011, Sumatra 2004) y un
modelo que no las acierta parece peor que predecir cero.

Con `--horizons-csv` repite la comparación sobre la tabla larga: ahí se predice
lo que falta entre `t` y `T` sabiendo lo observado hasta `t`.

Uso:
    python scripts/feature_baseline.py clusters_combined_....csv
    python scripts/feature_baseline.py clusters_combined_....csv \\
        --horizons-csv horizons_combined_....csv --time-split
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

TARGET = "n_aftershocks"

# Lo que ya describe al sismo principal sin mirar la secuencia.
CORE = (
    "mainshock_mag",
    "mainshock_depth",
    "mainshock_lat",
    "mainshock_lon",
    "dist_to_border_km",
)

# El bloque que se quiere evaluar.
PRIOR = (
    "n_prior_2d",
    "n_prior_30d",
    "n_prior_365d",
    "prior_max_mag_365d",
    "days_since_prior",
    "prior_window_truncated",
)

COLUMNS = (
    "mainshock_mag",
    "mainshock_depth",
    "n_prior_2d",
    "n_prior_30d",
    "n_prior_365d",
    "prior_max_mag_365d",
    "days_since_prior",
    "dist_to_border_km",
)

MAG_BINS = [0, 4, 5, 6, 7, 10]
MAG_LABELS = ["<4", "4-5", "5-6", "6-7", "7+"]

# 0 / 1-4 / 5+: con la distribución real, entre 1 y 4 eventos ya separan un
# enjambre de una zona tranquila; más de 5 es una secuencia claramente en curso.
PRIOR_BINS = [-1, 0, 4, 10**6]
PRIOR_LABELS = ["0", "1-4", "5+"]

# Para la eta cuadrada el objetivo se parte en tramos: así deja de ser una
# numérica y pasa a ser los "muchos grupos" que pide esa medida.
TARGET_BINS = [-1, 0, 4, 19, 10**9]
TARGET_LABELS = ["0", "1-4", "5-19", "20+"]

# Features derivadas: no están en el csv, salen de combinar dos columnas. Se
# evalúan en el semáforo pero no entran al baseline, que es el del csv tal cual.
DERIVED = ("mag_excess", "mag_gap")

# Umbrales del semáforo: (corte rojo/amarillo, corte amarillo/verde).
CUTS = {
    "separacion": (0.2, 0.8),
    "eta2": (0.05, 0.25),
    "correlacion": (0.2, 0.6),
    "brecha": (0.05, 0.15),
}


def summary(df: pd.DataFrame) -> None:
    target = df[TARGET]
    print(f"filas: {len(df)}")
    print(
        "objetivo %s: media %.2f | mediana %.0f | p90 %.0f | max %d | en cero %.1f%%"
        % (
            TARGET,
            target.mean(),
            target.median(),
            target.quantile(0.9),
            target.max(),
            100 * (target == 0).mean(),
        )
    )
    if "root_is_mainshock" in df.columns:
        print(
            "el evento que abrio la secuencia es el principal en %.1f%% de las filas "
            "(en el resto, el mayor llego despues: la magnitud del principal tiene hindsight)"
            % (100 * df["root_is_mainshock"].mean())
        )


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman y Pearson sobre log1p, global y dentro de cada región."""
    rows = []
    for column in COLUMNS:
        if column not in df.columns:
            continue
        pair = df[[column, TARGET]].dropna()
        if len(pair) < 3 or pair[column].std() == 0:
            continue
        row = {
            "columna": column,
            "n": len(pair),
            "spearman": pair[column].corr(pair[TARGET], method="spearman"),
            "pearson_log": np.corrcoef(
                np.log1p(pair[column].clip(lower=0)), np.log1p(pair[TARGET])
            )[0, 1],
        }
        if "region" in df.columns:
            within = [
                group[column].corr(group[TARGET], method="spearman")
                for _, group in df.dropna(subset=[column]).groupby("region")
            ]
            row["spearman_regiones"] = np.nanmean(within)
        rows.append(row)
    return pd.DataFrame(rows).set_index("columna").round(3)


def _color(value: float, measure: str) -> str:
    """R, Y o G según los cortes del semáforo. Se mira el valor absoluto."""
    low, high = CUTS[measure]
    magnitude = abs(value)
    if magnitude < low:
        return "R"
    return "Y" if magnitude < high else "G"


def _separation(values: pd.Series, positive: pd.Series) -> float:
    """Diferencia de medias entre los dos grupos, en desvíos de la feature."""
    valid = values.notna()
    spread = values[valid].std()
    if not spread:
        return 0.0
    return float(
        (values[valid & positive].mean() - values[valid & ~positive].mean()) / spread
    )


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Proporción de la varianza de la feature explicada por los grupos."""
    valid = values.notna() & groups.notna()
    x, g = values[valid].astype(float), groups[valid]
    if x.empty or not x.std():
        return 0.0
    grand = x.mean()
    between = sum(
        len(part) * (part.mean() - grand) ** 2 for _, part in x.groupby(g, observed=True)
    )
    total = ((x - grand) ** 2).sum()
    return float(between / total) if total else 0.0


def traffic_light(df: pd.DataFrame) -> None:
    """El semáforo de asociación, feature por feature.

    - separacion estandarizada: la feature entre las dos clases de
      `has_aftershock` (numérica contra dos grupos);
    - eta cuadrada: la feature entre los tramos del objetivo
      (numérica contra muchos grupos);
    - correlacion: Spearman y Pearson crudo contra `n_aftershocks`;
    - brecha: `|Spearman - Pearson|`, que delata que la relación no es recta
      (un verde acá no es un logro: es "no uses Pearson crudo, mirá el rango").

    Se evalúan también `mag_excess` y `mag_gap`, que no vienen en el csv pero
    salen de combinar columnas. El veredicto es de asociación **marginal**: no
    ve interacciones (el caso de `n_prior_2d`) ni el aporte condicional dentro
    del modelo, que es lo que mide la importancia por permutación.
    """
    work = df.copy()
    if {"mainshock_mag", "region_mc"} <= set(work.columns):
        work["mag_excess"] = work["mainshock_mag"] - work["region_mc"]
    if {"mainshock_mag", "prior_max_mag_365d"} <= set(work.columns):
        work["mag_gap"] = work["mainshock_mag"] - work["prior_max_mag_365d"]

    target = work[TARGET].astype(float)
    positive = (
        work["has_aftershock"].astype(bool)
        if "has_aftershock" in work.columns
        else target > 0
    )
    groups = pd.cut(target, bins=TARGET_BINS, labels=TARGET_LABELS)

    rows = []
    for column in COLUMNS + DERIVED:
        if column not in work.columns:
            continue
        values = work[column].astype(float)
        pair = pd.DataFrame({"x": values, "y": target}).dropna()
        if len(pair) < 3 or not pair["x"].std():
            continue
        spearman = pair["x"].corr(pair["y"], method="spearman")
        pearson = pair["x"].corr(pair["y"])
        separation = _separation(values, positive)
        eta2 = _eta_squared(values, groups)
        gap = abs(spearman - pearson)
        rows.append(
            {
                "feature": column,
                "sep_est": f"{_color(separation, 'separacion')} {separation:6.3f}",
                "eta2": f"{_color(eta2, 'eta2')} {eta2:6.3f}",
                "spearman": f"{_color(spearman, 'correlacion')} {spearman:6.3f}",
                "pearson": f"{_color(pearson, 'correlacion')} {pearson:6.3f}",
                "brecha": f"{_color(gap, 'brecha')} {gap:6.3f}",
            }
        )

    print("\nsemaforo (R = rojo, Y = amarillo, G = verde):")
    print("  separacion (2 grupos)      R < 0.2   Y 0.2-0.8   G > 0.8")
    print("  eta cuadrada (4 tramos)    R < 0.05  Y 0.05-0.25 G > 0.25")
    print("  correlacion                R < 0.2   Y 0.2-0.6   G > 0.6")
    print("  brecha Spearman-Pearson    R < 0.05  Y 0.05-0.15 G > 0.15")
    print()
    print(pd.DataFrame(rows).set_index("feature").to_string())


def prior_table(df: pd.DataFrame) -> None:
    """Media y proporción con réplicas por magnitud x sismicidad previa (2 d)."""
    window = df.dropna(subset=["mainshock_mag", "n_prior_2d"]).copy()
    window["mag_bin"] = pd.cut(
        window["mainshock_mag"], bins=MAG_BINS, labels=MAG_LABELS, right=False
    )
    window["prior_bin"] = pd.cut(
        window["n_prior_2d"], bins=PRIOR_BINS, labels=PRIOR_LABELS
    )
    window["tiene"] = window[TARGET] > 0

    mean = window.pivot_table(
        index="mag_bin", columns="prior_bin", values=TARGET,
        aggfunc="mean", observed=False,
    )
    share = window.pivot_table(
        index="mag_bin", columns="prior_bin", values="tiene",
        aggfunc="mean", observed=False,
    ) * 100
    count = window.pivot_table(
        index="mag_bin", columns="prior_bin", values=TARGET,
        aggfunc="size", observed=False,
    )

    print("\nmedia de %s por magnitud x n_prior_2d:" % TARGET)
    print(mean.round(2).to_string())
    print("\n% de secuencias con al menos una replica:")
    print(share.round(1).to_string())
    print("\nfilas por celda (para no leer de mas las celdas chicas):")
    print(count.fillna(0).astype(int).to_string())


def design(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Matriz numérica: las columnas pedidas más una dummy por región."""
    parts = [df[list(columns)].astype(float)]
    if "region" in df.columns:
        parts.append(pd.get_dummies(df["region"], prefix="region", dtype=float))
    return pd.concat(parts, axis=1)


def scores(y_true_log, pred_log, y_true) -> dict:
    return {
        "R2": r2_score(y_true_log, pred_log),
        "MAE_log": mean_absolute_error(y_true_log, pred_log),
        "MAE_replicas": mean_absolute_error(y_true, np.expm1(pred_log).clip(0)),
    }


def evaluate(df: pd.DataFrame, columns, time_split: bool) -> dict:
    y = df[TARGET].astype(float).to_numpy()
    y_log = np.log1p(y)
    X = design(df, columns)
    model = HistGradientBoostingRegressor(max_depth=6, random_state=0)
    if time_split:
        # Se entrena con las secuencias viejas y se testea con las nuevas: es lo
        # más parecido a predecir el próximo sismo que a interpolar en el medio.
        order = df.sort_values("mainshock_time").index.to_numpy()
        cut = int(0.7 * len(df))
        train, test = order[:cut], order[cut:]
        model.fit(X.loc[train], y_log[train])
        return scores(y_log[test], model.predict(X.loc[test]), y[test])
    pred = cross_val_predict(model, X, y_log, cv=KFold(5, shuffle=True, random_state=0))
    return scores(y_log, pred, y)


def model_report(df: pd.DataFrame, time_split: bool) -> None:
    y = df[TARGET].astype(float).to_numpy()
    y_log = np.log1p(y)
    rows = [
        {
            "modelo": "trivial: mediana (0 replicas)",
            **scores(y_log, np.zeros_like(y_log), y),
        },
        {
            "modelo": "trivial: media de log1p",
            **scores(y_log, np.full_like(y_log, y_log.mean()), y),
        },
        {
            "modelo": "magnitud + profundidad + posicion + region",
            **evaluate(df, CORE, time_split),
        },
        {
            "modelo": "+ sismicidad previa",
            **evaluate(df, CORE + PRIOR, time_split),
        },
    ]
    how = "corte temporal 70/30" if time_split else "5-fold aleatorio"
    print(f"\nbaseline del modelo ({how}, objetivo log1p):")
    print(pd.DataFrame(rows).set_index("modelo").round(3).to_string())


def importance_report(df: pd.DataFrame) -> None:
    y_log = np.log1p(df[TARGET].astype(float).to_numpy())
    X = design(df, CORE + PRIOR)
    model = HistGradientBoostingRegressor(max_depth=6, random_state=0).fit(X, y_log)
    result = permutation_importance(
        model, X, y_log, n_repeats=3, random_state=0,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(result.importances_mean)[::-1]
    print("\nimportancia (cuanto empeora el MAE al permutar la columna):")
    for i in order:
        print("  %-24s %.4f" % (X.columns[i], result.importances_mean[i]))


def horizons_report(clusters: pd.DataFrame, horizons: pd.DataFrame) -> None:
    """La misma pregunta con instante de predicción: qué falta entre t y T."""
    if "region" not in horizons.columns:
        print("\nhorizons: falta la columna 'region', se saltea.")
        return

    merged = horizons.merge(
        clusters[["cluster_id", "mainshock_mag", "mainshock_depth"]],
        on="cluster_id",
        how="left",
    )

    print("\nhorizontes: relacion entre lo observado hasta t y lo que falta")
    print(
        pd.DataFrame(
            [
                {
                    "horizonte_h": hour,
                    "spearman_n_aft_t": group["n_aft_t"].corr(
                        group["y_aft"], method="spearman"
                    ),
                    "media_y_aft": group["y_aft"].mean(),
                    "pct_sin_replicas": 100 * (group["y_aft"] == 0).mean(),
                }
                for hour, group in merged.groupby("horizon_hours")
            ]
        )
        .set_index("horizonte_h")
        .round(3)
        .to_string()
    )

    at = merged[merged["horizon_hours"] == merged["horizon_hours"].min()]
    rows = []
    y = at["y_aft"].astype(float).to_numpy()
    y_log = np.log1p(y)
    model = HistGradientBoostingRegressor(max_depth=6, random_state=0)
    for label, columns in (
        ("magnitud + profundidad + region", ["mainshock_mag", "mainshock_depth"]),
        (
            "+ observado hasta t",
            ["mainshock_mag", "mainshock_depth", "n_aft_t", "max_aft_mag_t",
             "hours_since_last_t"],
        ),
    ):
        X = design(at, columns)
        pred = cross_val_predict(model, X, y_log, cv=KFold(5, shuffle=True, random_state=0))
        rows.append({"modelo": label, **scores(y_log, pred, y)})
    print(
        "\nobjetivo y_aft en el horizonte de %gh (5-fold):"
        % at["horizon_hours"].iloc[0]
    )
    print(pd.DataFrame(rows).set_index("modelo").round(3).to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clusters_csv", help="csv de secuencias (clusters_*.csv)")
    parser.add_argument("--horizons-csv", help="tabla larga de horizontes")
    parser.add_argument(
        "--time-split",
        action="store_true",
        help="evalúa con las secuencias viejas de train y las nuevas de test",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.clusters_csv).reset_index(drop=True)
    if args.time_split:
        df["mainshock_time"] = pd.to_datetime(df["mainshock_time"], format="ISO8601", utc=True)

    summary(df)
    print("\ncorrelacion con el objetivo:")
    print(correlations(df).to_string())
    traffic_light(df)
    prior_table(df)
    model_report(df, args.time_split)
    importance_report(df)

    if args.horizons_csv:
        horizons_report(df, pd.read_csv(args.horizons_csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
