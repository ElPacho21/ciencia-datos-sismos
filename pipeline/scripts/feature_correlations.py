"""Correlación de cada feature causal con el objetivo, sólo números.

Para cada columna candidata calcula, sobre las secuencias **no censuradas**
(`mainshock_time + T <= endtime`):

- Pearson y Spearman contra `n_aftershocks`;
- la brecha contra `has_aftershock`: media de la feature en las secuencias con
  réplicas menos la media en las que no, en desvíos típicos de la feature.

Pearson y Spearman juntos importan: un Pearson alto con un Spearman bajo suele
ser un puñado de secuencias extremas (un enjambre) tirando de la recta.

Uso:
    python scripts/feature_correlations.py clusters_combined_....csv --endtime 2026-01-01
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

FEATURES = (
    "mainshock_mag",
    "mainshock_depth",
    "n_prior_2d",
    "n_prior_30d",
    "n_prior_365d",
    "prior_max_mag_365d",
    "days_since_prior",
    "dist_to_border_km",
)


def correlations(df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    rows = []
    positive = df["n_aftershocks"] > 0
    for column in features:
        if column not in df.columns:
            continue
        x = df[column].astype(float)
        valid = x.notna()
        if valid.sum() < 3 or x[valid].std() == 0:
            continue
        gap = (x[valid & positive].mean() - x[valid & ~positive].mean()) / x[valid].std()
        rows.append(
            {
                "feature": column,
                "n": int(valid.sum()),
                "pearson": x[valid].corr(df.loc[valid, "n_aftershocks"].astype(float)),
                "spearman": x[valid].corr(
                    df.loc[valid, "n_aftershocks"].astype(float), method="spearman"
                ),
                "gap_std": gap,
            }
        )
    return pd.DataFrame(rows).set_index("feature").round(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clusters_csv")
    parser.add_argument("--endtime", required=True)
    parser.add_argument("--horizon-days", type=float, default=30)
    parser.add_argument("--by-region", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.clusters_csv)
    df["mainshock_time"] = pd.to_datetime(df["mainshock_time"], format="ISO8601", utc=True)
    end = pd.Timestamp(args.endtime, tz="UTC")
    censored = df["mainshock_time"] + pd.Timedelta(days=args.horizon_days) > end
    kept = df[~censored]
    print(f"rows: {len(df)} | censored: {int(censored.sum())} | used: {len(kept)}")

    print("\nAll rows used:")
    print(correlations(kept).to_string())
    if args.by_region and "region" in kept.columns:
        for name, group in kept.groupby("region"):
            print(f"\n{name} ({len(group)} rows):")
            print(correlations(group).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
