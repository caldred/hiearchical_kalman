#!/usr/bin/env python3
import argparse
import os
import polars as pl


METRICS_COLS = [
    "pitch_id_cubs",
    "batter_id_cubs",
    "game_date",
    "bat_length",
    "swing_vertical_bat_angle",
    "attack_angle",
    "max_bat_speed",
]

DAMAGE_COLS = [
    "pitch_id_cubs",
    "hard_hit_velocity",
    "launch_angle_value",
    "damage_value",
]

DAMAGE_REQUIRED_COLS = DAMAGE_COLS + ["batter_id_cubs", "game_date"]

VALUE_COLS = [
    "bat_length",
    "swing_vertical_bat_angle",
    "attack_angle",
    "max_bat_speed",
    "hard_hit_velocity",
    "launch_angle_value",
    "damage_value",
]


def _require_columns(df: pl.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{label} is missing required columns: {missing_str}")


def _to_timestamp(col: pl.Expr) -> pl.Expr:
    # Parse dates that may arrive as strings or dates.
    return col.cast(pl.Utf8).str.to_datetime(strict=False).alias("timestamp")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine two pitch-level CSV extracts into a long-format parquet."
    )
    parser.add_argument("--metrics-csv", required=True, help="CSV from pitch_cubs + batter_checklist_metrics")
    parser.add_argument("--damage-csv", required=True, help="CSV from pitch_cubs + pitch_decomposition + batted_ball")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument(
        "--scaling-output",
        default=None,
        help="Optional path to write per-skill scaling factors (CSV).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.metrics_csv):
        raise FileNotFoundError(f"Missing file: {args.metrics_csv}")
    if not os.path.isfile(args.damage_csv):
        raise FileNotFoundError(f"Missing file: {args.damage_csv}")

    metrics = pl.read_csv(args.metrics_csv)
    damage = pl.read_csv(args.damage_csv)

    _require_columns(metrics, METRICS_COLS, "metrics CSV")
    _require_columns(damage, DAMAGE_REQUIRED_COLS, "damage CSV")

    damage = damage.select(DAMAGE_REQUIRED_COLS)

    joined = metrics.join(damage, on="pitch_id_cubs", how="full", suffix="_right")

    joined = (
        joined
        .with_columns(
            pl.coalesce([pl.col("batter_id_cubs"), pl.col("batter_id_cubs_right")]).alias("batter_id_cubs"),
            pl.coalesce([pl.col("game_date"), pl.col("game_date_right")]).alias("game_date"),
        )
        .drop(["batter_id_cubs_right", "game_date_right"])
    )

    missing_meta = joined.filter(
        pl.col("batter_id_cubs").is_null() | pl.col("game_date").is_null()
    ).height
    if missing_meta > 0:
        raise ValueError(
            f"Found {missing_meta} rows without batter_id_cubs or game_date after join."
        )

    long_df = (
        joined
        .unpivot(
            index=["batter_id_cubs", "game_date"],
            on=VALUE_COLS,
            variable_name="skill_name",
            value_name="observed_value",
        )
        .drop_nulls(["observed_value"])
        .with_columns(
            _to_timestamp(pl.col("game_date")),
            pl.col("batter_id_cubs").cast(pl.Int64).alias("player_id"),
            pl.col("observed_value").cast(pl.Float64, strict=False),
        )
        .select(["player_id", "timestamp", "skill_name", "observed_value"])
    )

    scaling = (
        long_df
        .group_by("skill_name")
        .agg(
            pl.col("observed_value").mean().alias("mean"),
            pl.col("observed_value").std().alias("std"),
        )
        .with_columns(
            pl.when(pl.col("std").is_null() | (pl.col("std") <= 0))
            .then(pl.lit(1.0))
            .otherwise(pl.col("std"))
            .alias("std")
        )
    )

    long_df = (
        long_df
        .join(scaling, on="skill_name", how="left")
        .with_columns(
            ((pl.col("observed_value") - pl.col("mean")) / pl.col("std")).alias("observed_value")
        )
        .select(["player_id", "timestamp", "skill_name", "observed_value"])
    )

    scaling_path = args.scaling_output or f"{args.output}.scaling.csv"
    scaling.write_csv(scaling_path)
    long_df.write_parquet(args.output)
    print(f"Wrote {long_df.height} rows to {args.output}")
    print(f"Wrote scaling factors to {scaling_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
