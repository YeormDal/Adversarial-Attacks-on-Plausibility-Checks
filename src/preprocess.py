import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # handled in load_config()


DEFAULT_CONFIG: Dict[str, object] = {
    "resample_hz": 10,
    "data_root": "data",
    "output_path": "outputs/veremi_10hz.parquet",
    "jerk_median_filter": True,
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(path: Path) -> Dict[str, object]:
    config = DEFAULT_CONFIG.copy()
    if not path.exists():
        logging.warning("Config file %s not found; using defaults.", path)
        return config
    if yaml is None:
        logging.warning("pyyaml not installed; cannot parse YAML. Using defaults.")
        return config
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            for key, value in loaded.items():
                if value is not None:
                    config[key] = value
        logging.info("Loaded config from %s", path)
    except Exception as exc:
        logging.exception("Failed to read config %s: %s", path, exc)
    return config


def find_csv_files(root: Path) -> List[Path]:
    if not root.exists():
        logging.warning("data_root %s does not exist.", root)
        return []
    files = sorted(root.glob("*.csv"))
    logging.info("Found %d CSV file(s) under %s", len(files), root)
    return files


def load_csvs(files: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                logging.warning("File %s is empty; skipping.", csv_path)
                continue
            frames.append(df)
            logging.info("Loaded %s (%d rows)", csv_path.name, len(df))
        except Exception:
            logging.exception("Failed to read %s", csv_path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def generate_self_test() -> pd.DataFrame:
    logging.info("data_root empty; entering self-test mode with dummy data.")
    rows = []
    duration_seconds = 10
    hz = 1
    times = np.arange(0, duration_seconds, 1 / hz, dtype=float)
    for veh_id in (101, 102):
        speed = 5 + veh_id * 0.05
        accel = 0.1 * (veh_id % 3)
        for t in times:
            x = speed * t + 0.5 * accel * t**2 + veh_id
            y = veh_id * 0.02 * t
            rows.append(
                {
                    "t": float(t),
                    "id": int(veh_id),
                    "x": x,
                    "y": y,
                    "v": speed + accel * t,
                    "a": accel,
                    "r": np.nan,
                    "alat": np.nan,
                    "d": np.nan,
                }
            )
    return pd.DataFrame(rows)


def detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def resample_vehicle(df: pd.DataFrame, t_col: str, resample_hz: int) -> pd.DataFrame:
    df = df.copy()
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df = df.dropna(subset=[t_col]).sort_values(t_col)
    if df.empty:
        return df

    time_start = float(df[t_col].min())
    time_end = float(df[t_col].max())
    step = 1.0 / float(resample_hz)
    if time_end <= time_start:
        time_grid = np.array([time_start])
    else:
        time_grid = np.arange(time_start, time_end + step / 2.0, step)
        time_grid = np.round(time_grid, 9)  # mitigate float accumulation

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if t_col in numeric_cols:
        numeric_cols.remove(t_col)
    df_indexed = df.set_index(t_col)
    resampled = df_indexed.reindex(time_grid)

    if numeric_cols:
        resampled[numeric_cols] = resampled[numeric_cols].interpolate(
            method="index", limit_direction="both"
        )
    resampled = resampled.ffill().bfill()

    resampled = resampled.reset_index().rename(columns={"index": t_col})
    return resampled


def compute_kinematics(df: pd.DataFrame, t_col: str, resample_hz: int) -> pd.DataFrame:
    df = df.copy()
    dt = 1.0 / float(resample_hz)

    x_col = detect_column(df, ("x", "pos_x", "px", "longitude", "lon"))
    y_col = detect_column(df, ("y", "pos_y", "py", "latitude", "lat"))
    s_col = detect_column(df, ("s", "distance", "pos", "along_track"))
    v_col = detect_column(df, ("v", "speed"))
    a_col = detect_column(df, ("a", "accel", "acceleration"))

    if x_col and y_col:
        dx = np.gradient(df[x_col].to_numpy(dtype=float), dt, edge_order=2)
        dy = np.gradient(df[y_col].to_numpy(dtype=float), dt, edge_order=2)
        v = np.sqrt(dx**2 + dy**2)
    elif s_col:
        v = np.gradient(df[s_col].to_numpy(dtype=float), dt, edge_order=2)
    elif v_col:
        v = pd.Series(df[v_col], dtype=float).interpolate().ffill().bfill().to_numpy()
    else:
        v = np.full(len(df), np.nan, dtype=float)

    if a_col and not np.any(np.isnan(df[a_col].to_numpy(dtype=float))):
        a = df[a_col].to_numpy(dtype=float)
    else:
        a = np.gradient(v, dt, edge_order=2)

    jerk = np.gradient(a, dt, edge_order=2)

    df["v"] = v
    df["a"] = a
    df["jerk"] = jerk
    return df


def apply_jerk_filter(df: pd.DataFrame) -> pd.DataFrame:
    if "jerk" not in df:
        return df
    series = pd.Series(df["jerk"].to_numpy(dtype=float))
    filtered = series.rolling(window=3, center=True, min_periods=1).median()
    df = df.copy()
    df["jerk"] = filtered.to_numpy(dtype=float)
    return df


def process_dataframe(raw: pd.DataFrame, resample_hz: int, jerk_filter: bool) -> pd.DataFrame:
    if raw.empty:
        return raw

    t_col = detect_column(raw, ("t", "time", "timestamp")) or "t"
    id_col = detect_column(raw, ("id", "veh_id", "vehicle_id")) or "id"

    if t_col not in raw.columns:
        logging.warning("Time column missing; creating synthetic 't' from index.")
        raw = raw.reset_index().rename(columns={"index": "t"})
        t_col = "t"
    if id_col not in raw.columns:
        logging.warning("ID column missing; assigning id=1 for all rows.")
        raw[id_col] = 1

    processed_chunks: List[pd.DataFrame] = []
    vehicle_ids = sorted(raw[id_col].dropna().unique())
    logging.info("Processing %d vehicle(s).", len(vehicle_ids))

    for veh_id in vehicle_ids:
        veh_rows = raw[raw[id_col] == veh_id]
        if veh_rows.empty:
            continue
        resampled = resample_vehicle(veh_rows, t_col, resample_hz)
        if resampled.empty:
            continue
        resampled[id_col] = veh_id
        resampled = compute_kinematics(resampled, t_col, resample_hz)
        if jerk_filter:
            resampled = apply_jerk_filter(resampled)
        processed_chunks.append(resampled)

    if not processed_chunks:
        return pd.DataFrame()

    result = pd.concat(processed_chunks, ignore_index=True, sort=False)
    ordered_cols = [t_col, id_col] + [c for c in result.columns if c not in (t_col, id_col)]
    return result[ordered_cols]


def save_dataframe(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(destination, index=False, engine="pyarrow")
        logging.info("Saved output to %s (parquet).", destination)
    except Exception as exc:
        logging.warning("Failed to write parquet (%s); falling back to CSV.", exc)
        fallback = destination.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        logging.info("Saved output to %s (csv).", fallback)


def summarize(df: pd.DataFrame, id_col: str) -> None:
    if df.empty:
        logging.info("Resulting dataframe is empty.")
        return
    cols = list(df.columns)
    vehicles = int(df[id_col].nunique()) if id_col in df else 0
    frames = len(df)
    logging.info("Columns: %s", cols)
    logging.info("Number of vehicles: %d", vehicles)
    logging.info("Total frames: %d", frames)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(5).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample VeReMi CSV logs, recompute kinematics, and export Parquet."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    resample_hz = int(config.get("resample_hz", DEFAULT_CONFIG["resample_hz"]))
    data_root = Path(config.get("data_root", DEFAULT_CONFIG["data_root"]))
    output_path = Path(config.get("output_path", DEFAULT_CONFIG["output_path"]))
    jerk_filter = bool(config.get("jerk_median_filter", DEFAULT_CONFIG["jerk_median_filter"]))

    logging.info(
        "Configuration: resample_hz=%d | data_root=%s | output_path=%s | jerk_median_filter=%s",
        resample_hz,
        data_root,
        output_path,
        jerk_filter,
    )

    csv_files = find_csv_files(data_root)
    if csv_files:
        raw_df = load_csvs(csv_files)
    else:
        raw_df = generate_self_test()

    if raw_df.empty:
        logging.error("No data available for processing; exiting.")
        return

    processed = process_dataframe(raw_df, resample_hz, jerk_filter)
    if processed.empty:
        logging.error("Processing produced no rows; exiting.")
        return

    save_dataframe(processed, output_path)

    id_col = detect_column(processed, ("id", "veh_id", "vehicle_id")) or "id"
    summarize(processed, id_col)


if __name__ == "__main__":
    main()