import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

from modelTraining import ImprovedLandslideANN
from modelTraining import MAX_SELECTED_FEATURES, select_features
from preprocessing import fit_transform_preprocessor, transform_with_preprocessor
from project_paths import (
    BENCHMARK_RESULTS_DIR,
    MODELS_DIR,
    ensure_project_dirs,
    resolve_raw_csvs,
)


SEED = 42
N_BLOCKS = 25
MIN_BLOCK_SIZE = 10
SCRIPT_DIR = Path(__file__).resolve().parent
ensure_project_dirs()
MODEL_PACKAGE_PATH = MODELS_DIR / "landslide_model_advanced_complete.pth"
ANN_BATCH_SIZE = 128
ANN_EPOCHS = 80
ANN_PATIENCE = 10


def resolve_csvs():
    return resolve_raw_csvs()


def create_spatial_blocks(coords, n_blocks=25):
    n_side = int(np.sqrt(n_blocks))
    x_bins = np.linspace(coords["xcoord"].min(), coords["xcoord"].max(), n_side + 1)
    y_bins = np.linspace(coords["ycoord"].min(), coords["ycoord"].max(), n_side + 1)
    x_block = np.digitize(coords["xcoord"], x_bins) - 1
    y_block = np.digitize(coords["ycoord"], y_bins) - 1
    x_block = np.clip(x_block, 0, n_side - 1)
    y_block = np.clip(y_block, 0, n_side - 1)
    return y_block * n_side + x_block


def find_balanced_group_split(X, y, groups, test_size, seed_start=SEED, trials=200):
    target_rate = float(np.mean(y))
    best = None
    best_score = float("inf")

    for offset in range(trials):
        splitter = GroupShuffleSplit(
            test_size=test_size,
            n_splits=1,
            random_state=seed_start + offset,
        )
        train_idx, val_idx = next(splitter.split(X, y, groups=groups))
        train_rate = float(np.mean(y.iloc[train_idx]))
        val_rate = float(np.mean(y.iloc[val_idx]))
        score = abs(train_rate - target_rate) + abs(val_rate - target_rate)
        if score < best_score:
            best_score = score
            best = (train_idx, val_idx, seed_start + offset)

    return best


def compute_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "auroc": roc_auc_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, probabilities) if len(np.unique(y_true)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y_true, predictions) if len(np.unique(predictions)) > 1 else 0.0,
        "brier": brier_score_loss(y_true, probabilities),
        "threshold": threshold,
    }


def tune_threshold(model, X_train, y_train, groups_train):
    inner_train_idx, inner_val_idx, inner_seed = find_balanced_group_split(
        X_train,
        y_train,
        groups_train,
        test_size=0.25,
        seed_start=SEED + 1000,
        trials=100,
    )

    X_inner_train = X_train.iloc[inner_train_idx]
    y_inner_train = y_train.iloc[inner_train_idx]
    X_inner_val = X_train.iloc[inner_val_idx]
    y_inner_val = y_train.iloc[inner_val_idx]

    model_name = model.__class__.__name__
    if model_name == "GradientBoostingClassifier":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_inner_train)
        model.fit(X_inner_train, y_inner_train, sample_weight=sample_weight)
    else:
        model.fit(X_inner_train, y_inner_train)

    val_probabilities = model.predict_proba(X_inner_val)[:, 1]
    thresholds = np.linspace(0.20, 0.80, 61)
    scored = [
        compute_metrics(y_inner_val.to_numpy(), val_probabilities, float(threshold))
        for threshold in thresholds
    ]
    best = max(scored, key=lambda item: (item["mcc"], item["f1"], item["pr_auc"]))
    return best["threshold"], inner_seed


def make_models():
    return {
        "RF": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=SEED,
        ),
    }


def train_ann_with_inner_validation(X_train, y_train, X_val, y_val):
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=ANN_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=ANN_BATCH_SIZE,
        shuffle=False,
    )

    class_counts = np.bincount(y_train.to_numpy())
    pos_weight_value = class_counts[0] / max(class_counts[1], 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    model = ImprovedLandslideANN(X_train.shape[1])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for _ in range(ANN_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                total_val_loss += criterion(model(X_batch), y_batch).item()
        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= ANN_PATIENCE:
            break

    model.load_state_dict(best_state)

    with torch.no_grad():
        val_probabilities = torch.sigmoid(model(X_val_tensor)).cpu().numpy().flatten()
    thresholds = np.linspace(0.20, 0.80, 61)
    scored = [
        compute_metrics(y_val.to_numpy(), val_probabilities, float(threshold))
        for threshold in thresholds
    ]
    best = max(scored, key=lambda item: (item["mcc"], item["f1"], item["pr_auc"]))
    return model, scaler, best["threshold"]


def evaluate_ann_on_fold(X_train, y_train, train_groups, X_test, y_test):
    inner_train_idx, inner_val_idx, inner_seed = find_balanced_group_split(
        X_train,
        y_train,
        train_groups,
        test_size=0.25,
        seed_start=SEED + 1000,
        trials=100,
    )

    model, scaler, threshold = train_ann_with_inner_validation(
        X_train.iloc[inner_train_idx],
        y_train.iloc[inner_train_idx],
        X_train.iloc[inner_val_idx],
        y_train.iloc[inner_val_idx],
    )

    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()

    metrics = compute_metrics(y_test.to_numpy(), probabilities, threshold)
    return metrics, inner_seed


def main():
    landslide_csv, non_landslide_csv = resolve_csvs()

    landslides = pd.read_csv(landslide_csv)
    non_landslides = pd.read_csv(non_landslide_csv)
    landslides["label"] = 1
    non_landslides["label"] = 0

    full_data = pd.concat([landslides, non_landslides], ignore_index=True)
    full_data = full_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_raw = full_data.drop(columns=["label"]).replace({True: 1, False: 0})
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = full_data["label"].astype(int)
    coordinates = X_raw[["xcoord", "ycoord"]].copy()
    block_ids = create_spatial_blocks(coordinates, n_blocks=N_BLOCKS)
    block_summary = pd.DataFrame({"block": block_ids, "label": y}).groupby("block")["label"].agg(["count", "mean"])
    valid_blocks = block_summary[block_summary["count"] >= MIN_BLOCK_SIZE].index.to_numpy()
    valid_mask = np.isin(block_ids, valid_blocks)

    X_raw = X_raw.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    block_ids = block_ids[valid_mask]
    coordinates = coordinates.loc[valid_mask].reset_index(drop=True)

    print(f"Benchmark dataset shape: {X_raw.shape}")
    print("Using fold-specific training-only preprocessing and feature selection")
    print(f"Using {len(np.unique(block_ids))} populated spatial blocks")

    fold_assignment_rows = []
    detailed_rows = []
    fold_feature_rows = []
    gkf = GroupKFold(n_splits=5)

    for fold_index, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y, groups=block_ids), start=1):
        X_train_raw = X_raw.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test_raw = X_raw.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        train_groups = block_ids[train_idx]
        test_groups = block_ids[test_idx]

        print(f"\nFold {fold_index}/5")
        print(f"  Train samples: {len(X_train_raw):,}, positive rate={y_train.mean():.3f}")
        print(f"  Test samples:  {len(X_test_raw):,}, positive rate={y_test.mean():.3f}")

        for block in np.unique(test_groups):
            fold_assignment_rows.append({"fold": fold_index, "block": int(block)})

        X_train_engineered, preprocessor = fit_transform_preprocessor(X_train_raw)
        X_test_engineered = transform_with_preprocessor(X_test_raw, preprocessor)
        selected_features, _ = select_features(
            X_train_engineered,
            y_train,
            max_features=MAX_SELECTED_FEATURES,
        )
        X_train = X_train_engineered[selected_features].copy()
        X_test = X_test_engineered[selected_features].copy()

        for feature in selected_features:
            fold_feature_rows.append({"fold": fold_index, "feature": feature})

        ann_metrics, ann_inner_seed = evaluate_ann_on_fold(X_train, y_train, train_groups, X_test, y_test)
        ann_metrics["model"] = "ANN"
        ann_metrics["fold"] = fold_index
        ann_metrics["n_test"] = len(X_test)
        ann_metrics["positive_rate_test"] = float(y_test.mean())
        detailed_rows.append(ann_metrics)
        print(
            f"  ANN: tuned threshold={ann_metrics['threshold']:.2f} using inner split seed {ann_inner_seed}"
        )
        print(
            f"    AUROC={ann_metrics['auroc']:.3f} | PR-AUC={ann_metrics['pr_auc']:.3f} | "
            f"Acc={ann_metrics['accuracy']:.3f} | F1={ann_metrics['f1']:.3f} | MCC={ann_metrics['mcc']:.3f}"
        )

        for model_name, model in make_models().items():
            threshold, inner_seed = tune_threshold(model, X_train, y_train, train_groups)
            print(f"  {model_name}: tuned threshold={threshold:.2f} using inner split seed {inner_seed}")

            if model_name == "GB":
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
                model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train)

            probabilities = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test.to_numpy(), probabilities, threshold)
            metrics["model"] = model_name
            metrics["fold"] = fold_index
            metrics["n_test"] = len(X_test)
            metrics["positive_rate_test"] = float(y_test.mean())
            detailed_rows.append(metrics)

            print(
                f"    AUROC={metrics['auroc']:.3f} | PR-AUC={metrics['pr_auc']:.3f} | "
                f"Acc={metrics['accuracy']:.3f} | F1={metrics['f1']:.3f} | MCC={metrics['mcc']:.3f}"
            )

    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = (
        detailed_df.groupby("model")[["auroc", "pr_auc", "accuracy", "precision", "recall", "f1", "mcc", "brier", "threshold"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    fold_assignment_df = pd.DataFrame(fold_assignment_rows).sort_values(["fold", "block"])
    fold_features_df = pd.DataFrame(fold_feature_rows).sort_values(["fold", "feature"])
    detailed_path = BENCHMARK_RESULTS_DIR / "benchmark_results_detailed.csv"
    summary_path = BENCHMARK_RESULTS_DIR / "benchmark_results_summary.csv"
    folds_path = BENCHMARK_RESULTS_DIR / "benchmark_fold_assignments.csv"
    fold_features_path = BENCHMARK_RESULTS_DIR / "benchmark_fold_features.csv"
    metadata_path = BENCHMARK_RESULTS_DIR / "benchmark_metadata.json"

    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path)
    fold_assignment_df.to_csv(folds_path, index=False)
    fold_features_df.to_csv(fold_features_path, index=False)

    metadata = {
        "feature_source": "training_only_selection_per_outer_fold",
        "max_selected_features": MAX_SELECTED_FEATURES,
        "predictor_family": "12 continuous predictors + one-hot lithology + one-hot soil",
        "landslide_csv": str(landslide_csv),
        "non_landslide_csv": str(non_landslide_csv),
        "n_blocks_requested": N_BLOCKS,
        "n_blocks_populated": int(len(np.unique(block_ids))),
        "min_block_size": MIN_BLOCK_SIZE,
        "models": ["ANN"] + list(make_models().keys()),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nSaved benchmark outputs:")
    print(f"  Detailed fold metrics: {detailed_path}")
    print(f"  Summary metrics:       {summary_path}")
    print(f"  Fold assignments:      {folds_path}")
    print(f"  Fold features:         {fold_features_path}")
    print(f"  Metadata:              {metadata_path}")


if __name__ == "__main__":
    main()
