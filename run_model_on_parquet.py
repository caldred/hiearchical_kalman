#!/usr/bin/env python3
import argparse
import polars as pl

from model import HierarchicalKalmanFilter


def build_dag():
    return {
        "hard_hit_velocity": ["max_bat_speed", "bat_length"],
        "damage_value": ["hard_hit_velocity", "launch_angle_value", "attack_angle"],
        "launch_angle_value": ["swing_vertical_bat_angle", "attack_angle"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit HierarchicalKalmanFilter on long-format parquet input."
    )
    parser.add_argument("--input", required=True, help="Parquet path from prepare_input_parquet.py")
    parser.add_argument("--bin-size-days", type=int, default=7, help="Time bin size in days")
    parser.add_argument("--max-iter", type=int, default=10, help="EM iterations")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs")
    args = parser.parse_args()

    df = pl.read_parquet(args.input)

    required = {"player_id", "timestamp", "skill_name", "observed_value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(missing)}")

    dag = build_dag()
    model = HierarchicalKalmanFilter(dag, bin_size_days=args.bin_size_days, n_jobs=args.n_jobs)

    model.fit(df, max_iter=args.max_iter, verbose=True)

    estimates = model.get_estimates()
    print(estimates.head(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
