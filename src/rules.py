"""VeReMi plausibility rule evaluator.

This script loads a 10 Hz VeReMi Parquet dataset, runs L1/L2 plausibility
checks per vehicle track, and stores the augmented result as Parquet.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required to load the configuration. Install it via "
        "`python -m pip install pyyaml`."
    ) from exc

try:
    import pyarrow  # noqa: F401  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyarrow is required for Parquet IO. Install it via "
        "`python -m pip install pyarrow`."
    ) from exc


CONFIG_PATH_DEFAULT = Path("configs/base.yaml")


def setup_logging() -> None:
    """Configure root logger for concise console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_config(path: Path) -> Dict[str, Dict[str, object]]:
    """Load YAML configuration and validate required plausibility fields."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "plausibility" not in config:
        raise KeyError("Missing 'plausibility' section in configuration.")
    required_keys = [
        "dt",
        "eps_v",
        "tau_v",
        "tau_a",
        "tau_j",
        "kappa",
        "col_time",
        "col_track",
        "col_v",
        "col_a",
        "col_r",
        "col_alat",
    ]
    missing = [k for k in required_keys if k not in config["plausibility"]]
    if missing:
        raise KeyError(f"Missing plausibility config keys: {missing}")
    return config


def load_dataframe(parquet_path: Path) -> pd.DataFrame:
    """Load dataset from Parquet, ensuring the file exists."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input Parquet file not found: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    if df.empty:
        raise ValueError(f"Input dataset {parquet_path} is empty.")
    return df


def detect_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    """Return the first column present in DataFrame that matches the candidate list."""
    lower_map = {col.lower(): col for col in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def ensure_lateral_quantities(df: pd.DataFrame, config: Dict[str, object]) -> pd.DataFrame:
    """Estimate yaw-rate (r) and lateral acceleration if missing/NaN using positions."""
    dt = float(config["dt"])
    col_track = str(config["col_track"])
    col_r = str(config["col_r"])
    col_alat = str(config["col_alat"])

    df = df.copy()
    if col_r not in df.columns:
        df[col_r] = np.nan
    if col_alat not in df.columns:
        df[col_alat] = np.nan

    x_candidates: List[str] = []
    y_candidates: List[str] = []
    if isinstance(config.get("col_x"), str):
        x_candidates.append(str(config["col_x"]))
    if isinstance(config.get("col_y"), str):
        y_candidates.append(str(config["col_y"]))
    x_candidates.extend(["x", "pos_x", "px", "longitude", "lon"])
    y_candidates.extend(["y", "pos_y", "py", "latitude", "lat"])

    x_col = detect_column(df, x_candidates)
    y_col = detect_column(df, y_candidates)

    if x_col is None or y_col is None:
        if df[col_r].isna().any() or df[col_alat].isna().any():
            logging.warning(
                "Unable to estimate lateral quantities: missing position columns."
            )
        return df

    logging.info(
        "Estimating lateral quantities using position columns '%s' and '%s'.",
        x_col,
        y_col,
    )

    _coerce_numeric(df, [x_col, y_col])
    grouped = df.groupby(col_track, sort=False)

    vx = grouped[x_col].diff().div(dt).fillna(0.0)
    vy = grouped[y_col].diff().div(dt).fillna(0.0)

    ax = vx.groupby(df[col_track], sort=False).diff().div(dt).fillna(0.0)
    ay = vy.groupby(df[col_track], sort=False).diff().div(dt).fillna(0.0)

    speed = np.hypot(vx, vy)
    speed_safe = speed.clip(lower=1e-6)
    alat_est = (vx * ay - vy * ax) / speed_safe
    low_speed_mask = speed < 1e-6
    alat_est = alat_est.where(~low_speed_mask, 0.0)
    alat_est = alat_est.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    heading = np.arctan2(vy, vx)
    heading = heading.groupby(df[col_track], sort=False).transform(
        lambda s: np.unwrap(s.to_numpy())
    )
    r_est = heading.groupby(df[col_track], sort=False).diff().div(dt).fillna(0.0)

    df[col_r] = df[col_r].fillna(r_est)
    df[col_alat] = df[col_alat].fillna(alat_est)

    if bool(config.get("r_in_deg", False)):
        logging.info("Converting r from degrees to radians as per configuration.")
        df[col_r] = np.deg2rad(df[col_r])
    if bool(config.get("alat_in_g", False)):
        logging.info("Converting a_lat from g to m/s^2 as per configuration.")
        df[col_alat] = df[col_alat] * 9.80665

    return df


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    """Cast specified columns to numeric dtype in-place."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _fill_groupwise(
    df: pd.DataFrame, group_col: str, target_cols: List[str]
) -> None:
    """Group-wise interpolate, then forward/backward fill for specified columns."""
    grouped = df.groupby(group_col, sort=False)
    for col in target_cols:
        df[col] = grouped[col].transform(lambda s: s.interpolate().ffill().bfill())


def compute_plausibility(
    df: pd.DataFrame, config: Dict[str, object]
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run L1/L2 plausibility computations and append diagnostic columns."""
    dt = float(config["dt"])
    eps_v = float(config["eps_v"])
    tau_v = float(config["tau_v"])
    tau_a = float(config["tau_a"])
    tau_j = float(config["tau_j"])
    kappa = float(config["kappa"])
    col_time = str(config["col_time"])
    col_track = str(config["col_track"])
    col_v = str(config["col_v"])
    col_a = str(config["col_a"])
    col_r = str(config["col_r"])
    col_alat = str(config["col_alat"])

    required_columns = [col_time, col_track, col_v, col_r, col_alat]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Input data missing required columns: {missing}")
    if col_a not in df.columns:
        df[col_a] = np.nan

    df = df.copy()
    df = df.sort_values([col_track, col_time]).reset_index(drop=True)

    _coerce_numeric(df, [col_v, col_a, col_r, col_alat])
    _fill_groupwise(df, col_track, [col_v, col_r, col_alat])

    grouped = df.groupby(col_track, sort=False)

    delta_v = grouped[col_v].diff().fillna(0.0)
    dv_dt = delta_v / dt

    accel = df[col_a].copy()
    accel = accel.where(~accel.isna(), dv_dt)
    df[col_a] = accel

    jerk = grouped[col_a].diff().fillna(0.0) / dt
    jerk = jerk.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    residual = (df[col_alat] - df[col_v] * df[col_r]).abs()

    abs_delta_v = delta_v.abs()
    abs_accel = df[col_a].abs()
    abs_jerk = jerk.abs()

    pass_L1 = (abs_delta_v <= tau_v) & (abs_accel <= tau_a) & (abs_jerk <= tau_j)
    is_first = grouped.cumcount() == 0
    pass_L2 = (residual <= kappa) | is_first

    failed_rule = np.select(
        [
            abs_delta_v > tau_v,
            abs_accel > tau_a,
            abs_jerk > tau_j,
            ~pass_L2,
        ],
        ["L1_v", "L1_a", "L1_j", "L2"],
        default="",
    )

    df["delta_v"] = delta_v
    df["jerk"] = jerk
    df["residual"] = residual
    df["pass_L1"] = pass_L1
    df["pass_L2"] = pass_L2
    df["failed_rule"] = failed_rule

    diagnostics = {
        "rows": len(df),
        "L1_pass_rate": float(pass_L1.mean()),
        "L2_pass_rate": float(pass_L2.mean()),
        "both_pass_rate": float((pass_L1 & pass_L2).mean()),
        "fail_counts": df.loc[df["failed_rule"] != "", "failed_rule"]
        .value_counts()
        .head(3)
        .to_dict(),
    }
    return df, diagnostics


def summarize(diagnostics: Dict[str, object]) -> None:
    """Log concise summary based on computed diagnostics."""
    logging.info("Total rows: %d", diagnostics["rows"])
    logging.info("L1 pass rate: %.4f", diagnostics["L1_pass_rate"])
    logging.info("L2 pass rate: %.4f", diagnostics["L2_pass_rate"])
    logging.info("Both pass rate: %.4f", diagnostics["both_pass_rate"])
    fail_counts = diagnostics["fail_counts"]
    if fail_counts:
        logging.info("Top-3 failure reasons:")
        for rule, count in fail_counts.items():
            logging.info("  %s: %d", rule, count)
    else:
        logging.info("No rule violations detected.")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run L1/L2 plausibility checks on VeReMi 10 Hz Parquet data."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        required=True,
        help="Path to input Parquet file produced by preprocessing.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        required=True,
        help="Path to write the augmented Parquet output.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        default=str(CONFIG_PATH_DEFAULT),
        help="Path to YAML configuration (default: configs/base.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args()

    config_path = Path(args.config_path)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    logging.info("Loading configuration from %s", config_path)
    config = load_config(config_path)
    plausibility_cfg = config["plausibility"]

    logging.info("Loading data from %s", input_path)
    df = load_dataframe(input_path)

    df = ensure_lateral_quantities(df, plausibility_cfg)

    logging.info("Computing plausibility diagnostics.")
    augmented_df, diagnostics = compute_plausibility(df, plausibility_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_df.to_parquet(output_path, index=False, engine="pyarrow")
    logging.info("Saved results to %s", output_path)

    summarize(diagnostics)


if __name__ == "__main__":
    main()