"""Train and evaluate misbehavior detectors on VeReMi plausibility-filtered data."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FeatureSplit = Tuple[np.ndarray, np.ndarray, np.ndarray]
LabelSplit = Tuple[np.ndarray, np.ndarray, np.ndarray]


def setup_logging() -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate misbehavior detection models on VeReMi data."
    )
    parser.add_argument(
        "--in",
        dest="clean_path",
        required=True,
        help="Path to plausibility-filtered clean frames parquet.",
    )
    parser.add_argument(
        "--attacks",
        dest="attack_path",
        default=None,
        help="Optional path to attack/adversarial parquet.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        required=True,
        help="YAML configuration path (expects 'seed' and 'split').",
    )
    parser.add_argument(
        "--outdir",
        dest="out_dir",
        required=True,
        help="Base directory for models/metrics/figures outputs.",
    )
    parser.add_argument(
        "--dump-splits",
        action="store_true",
        help="Persist boolean masks for train/val/test splits to split_masks.parquet.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, object]:
    """Load YAML configuration from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    if df.empty:
        raise ValueError(f"Parquet file {path} is empty.")
    return df


def add_label_column(df: pd.DataFrame, label: int) -> pd.DataFrame:
    """Return dataframe with a 'y' label column set to the provided value."""
    df = df.copy()
    df["y"] = label
    return df


def maybe_concat_attacks(clean_df: pd.DataFrame, attack_path: Optional[Path]) -> pd.DataFrame:
    """Concatenate clean and attack dataframes, creating labels as needed."""
    combined = add_label_column(clean_df, 0)
    if attack_path:
        attack_df = load_parquet(attack_path)
        attack_df = add_label_column(attack_df, 1)
        combined = pd.concat([combined, attack_df], ignore_index=True, sort=False)
        logging.info("Loaded attacks from %s (%d rows).", attack_path, len(attack_df))
    else:
        logging.info("No attacks provided; proceeding with clean data only.")
    return combined


def choose_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Select usable feature columns in priority order."""
    priority = ["v", "a_eff", "a", "jerk", "residual", "x", "y", "d", "r", "alat"]
    available: List[str] = []
    has_a_eff = "a_eff" in df.columns
    for feature in priority:
        if feature == "a" and has_a_eff:
            continue
        if feature in df.columns and feature not in available:
            available.append(feature)

    if not available:
        raise ValueError(
            "No valid feature columns found. Candidate features: "
            f"{priority}. Existing columns: {list(df.columns)}. "
            "Please update configs/base.yaml or preprocessing to expose at least one feature."
        )

    feature_df = df[available].copy()
    for col in available:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    feature_df = feature_df.fillna(0.0)
    return feature_df, available


def detect_scenario_column(df: pd.DataFrame) -> Optional[str]:
    """Return scenario column name if present."""
    candidates = ["S", "scenario", "scene"]
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def split_by_config_or_groups(
    df: pd.DataFrame, config: Dict[str, object], seed: int
) -> Dict[str, np.ndarray]:
    """Split dataframe indices into train/val/test based on config or group splits."""
    scenario_col = detect_scenario_column(df)
    if scenario_col and "split" in config:
        split_cfg = config["split"] or {}
        train_labels = [str(x) for x in split_cfg.get("train", [])]
        val_labels = [str(x) for x in split_cfg.get("val", [])]
        test_labels = [str(x) for x in split_cfg.get("test", [])]

        scen = df[scenario_col].astype(str)
        mask_train = scen.isin(train_labels)
        mask_val = scen.isin(val_labels)
        mask_test = scen.isin(test_labels)

        if not mask_train.any() or not mask_val.any() or not mask_test.any():
            raise ValueError(
                "Scenario-based split failed: check that train/val/test labels exist in data."
            )

        return {
            "train": df.index[mask_train].to_numpy(),
            "val": df.index[mask_val].to_numpy(),
            "test": df.index[mask_test].to_numpy(),
        }

    if "id" not in df.columns:
        raise KeyError("Column 'id' required for group-wise splitting is missing.")

    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    groups = df["id"].to_numpy()
    full_indices = np.arange(len(df))
    train_idx, rest_idx = next(gss.split(full_indices, df["y"].to_numpy(), groups=groups))

    if len(rest_idx) == 0:
        raise ValueError("Group split produced empty validation/test set.")

    remaining_groups = groups[rest_idx]
    remain_indices = full_indices[rest_idx]
    unique_groups = np.unique(remaining_groups)

    if unique_groups.size <= 1:
        midpoint = len(remain_indices) // 2
        val_idx = remain_indices[:midpoint]
        test_idx = remain_indices[midpoint:]
    else:
        gss_secondary = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
        val_idx, test_idx = next(
            gss_secondary.split(
                remain_indices,
                df["y"].to_numpy()[rest_idx],
                groups=remaining_groups,
            )
        )
        val_idx = remain_indices[val_idx]
        test_idx = remain_indices[test_idx]

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray],
) -> Dict[str, object]:
    """Compute precision/recall/f1/confusion (2x2) with per-class breakdown and ROC AUC."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "per_class": [
            {
                "label": int(label),
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
                "support": int(per_class_support[idx]),
            }
            for idx, label in enumerate([0, 1])
        ],
    }

    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def get_score(model, X: np.ndarray) -> Optional[np.ndarray]:
    """Return probability estimates or decision scores if available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            return scores
        if scores.shape[1] >= 2:
            return scores[:, 1]
    return None


def save_roc_figure(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fit_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    figures_dir: Path,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, object], Dict[str, object], pd.DataFrame]:
    """Train RF/MLP models, evaluate on validation, optionally plot ROC on test."""
    models: Dict[str, object] = {}

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    mlp.fit(X_train, y_train)
    models["mlp"] = mlp

    metrics: Dict[str, object] = {}
    val_prediction_rows: List[pd.DataFrame] = []

    for name, model in models.items():
        y_pred_val = model.predict(X_val)
        scores_val = get_score(model, X_val)
        metrics[name] = compute_metrics(y_val, y_pred_val, scores_val)

        score_series = scores_val if scores_val is not None else y_pred_val.astype(float)
        val_prediction_rows.append(
            pd.DataFrame(
                {
                    "model": name,
                    "y_true": y_val,
                    "y_pred": y_pred_val,
                    "score": score_series,
                    "split": "val",
                }
            )
        )

    val_predictions = pd.concat(val_prediction_rows, ignore_index=True) if val_prediction_rows else pd.DataFrame()

    test_has_two_classes = len(np.unique(y_test)) > 1
    if test_has_two_classes:
        for name, model in models.items():
            scores_test = get_score(model, X_test)
            if scores_test is None:
                continue
            figure_name = "rf" if name == "random_forest" else name
            save_roc_figure(
                y_true=y_test,
                y_scores=scores_test,
                out_path=figures_dir / f"roc_{figure_name}.png",
                title=f"ROC Curve ({name})",
            )

    rf_importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": models["random_forest"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    rf_importances.to_csv(figures_dir.parent / "rf_feature_importance.csv", index=False)

    mlp_clf = models["mlp"].named_steps["mlp"]
    first_layer = mlp_clf.coefs_[0]
    norms = np.linalg.norm(first_layer, axis=1)
    mlp_norms = pd.DataFrame(
        {
            "feature": feature_names,
            "l2_norm": norms,
            "first_layer_shape": [str(list(first_layer.shape))] * len(feature_names),
        }
    )
    mlp_norms.to_csv(figures_dir.parent / "mlp_feature_norms.csv", index=False)

    return models, metrics, val_predictions


def save_metrics(metrics: Dict[str, object], path: Path) -> None:
    """Persist metrics dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_feature_stats(
    scaler: StandardScaler, feature_names: Sequence[str], n_train: int, path: Path
) -> None:
    """Save feature statistics for single-class scenario."""
    stats = {
        "feature_names": list(feature_names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_train": int(n_train),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args()

    config = load_yaml(Path(args.config_path))
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    figures_dir = out_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    clean_df = load_parquet(Path(args.clean_path))
    attack_path = Path(args.attack_path) if args.attack_path else None
    combined_df = maybe_concat_attacks(clean_df, attack_path)

    features_df, feature_names = choose_features(combined_df)
    if "y" not in combined_df.columns:
        raise KeyError("Label column 'y' missing after concatenation.")

    split_indices = split_by_config_or_groups(combined_df, config, seed)

    if args.dump_splits:
        index_array = np.arange(len(combined_df))
        split_masks = pd.DataFrame(
            {
                "train": np.isin(index_array, split_indices["train"]),
                "val": np.isin(index_array, split_indices["val"]),
                "test": np.isin(index_array, split_indices["test"]),
            }
        )
        splits_path = models_dir / "split_masks.parquet"
        split_masks.to_parquet(splits_path, index=False)
        logging.info("Saved split masks to %s", splits_path)

    X_train = features_df.iloc[split_indices["train"]].to_numpy()
    y_train = combined_df.iloc[split_indices["train"]]["y"].to_numpy(dtype=int)
    X_val = features_df.iloc[split_indices["val"]].to_numpy()
    y_val = combined_df.iloc[split_indices["val"]]["y"].to_numpy(dtype=int)
    X_test = features_df.iloc[split_indices["test"]].to_numpy()
    y_test = combined_df.iloc[split_indices["test"]]["y"].to_numpy(dtype=int)

    unique_classes = np.unique(combined_df["y"].to_numpy(dtype=int))
    if unique_classes.size <= 1:
        logging.info("Single-class dataset detected; skipping supervised training.")
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, models_dir / "scaler.joblib")
        save_feature_stats(scaler, feature_names, len(X_train), models_dir / "feature_stats.json")
        save_metrics({"info": "single-class; no supervised metrics"}, models_dir / "metrics.json")
        return

    models, metrics, val_predictions = fit_and_eval(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        seed,
        figures_dir,
        feature_names,
    )

    if not val_predictions.empty:
        val_predictions.to_csv(models_dir / "val_predictions.csv", index=False)

    joblib.dump(models["random_forest"], models_dir / "rf.joblib")
    joblib.dump(models["mlp"], models_dir / "mlp.joblib")
    save_metrics(metrics, models_dir / "metrics.json")

    logging.info("Training and evaluation complete. Models saved to %s.", models_dir)


if __name__ == "__main__":
    main()