import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from project_paths import (
    MODELS_DIR,
    TRAINING_RESULTS_DIR,
    ensure_project_dirs,
    resolve_processed_csvs,
)


SEED = 42
MIN_BLOCK_SIZE = 10
N_BLOCKS = 25
MAX_SELECTED_FEATURES = 60
BATCH_SIZE = 128
NUM_EPOCHS = 120
PATIENCE = 15
SCRIPT_DIR = Path(__file__).resolve().parent
ensure_project_dirs()
BEST_MODEL_PATH = MODELS_DIR / "best_model_advanced.pth"
MODEL_PACKAGE_PATH = MODELS_DIR / "landslide_model_advanced_complete.pth"


torch.manual_seed(SEED)
np.random.seed(SEED)


def resolve_csvs():
    return resolve_processed_csvs()


def create_spatial_blocks(coords, n_blocks=25):
    n_side = int(np.sqrt(n_blocks))
    x_bins = np.linspace(coords["xcoord"].min(), coords["xcoord"].max(), n_side + 1)
    y_bins = np.linspace(coords["ycoord"].min(), coords["ycoord"].max(), n_side + 1)

    x_block = np.digitize(coords["xcoord"], x_bins) - 1
    y_block = np.digitize(coords["ycoord"], y_bins) - 1
    x_block = np.clip(x_block, 0, n_side - 1)
    y_block = np.clip(y_block, 0, n_side - 1)

    block_ids = y_block * n_side + x_block
    print(f"Created {len(np.unique(block_ids))} spatial blocks using a {n_side}x{n_side} grid")
    return block_ids


def find_balanced_group_split(X, y, groups, test_size, seed_start=SEED, trials=200):
    target_rate = float(np.mean(y))
    best_score = float("inf")
    best_split = None

    for offset in range(trials):
        splitter = GroupShuffleSplit(
            test_size=test_size,
            n_splits=1,
            random_state=seed_start + offset,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        train_rate = float(np.mean(y.iloc[train_idx]))
        test_rate = float(np.mean(y.iloc[test_idx]))
        size_balance = abs((len(test_idx) / len(y)) - test_size)
        score = abs(train_rate - target_rate) + abs(test_rate - target_rate) + size_balance

        if score < best_score:
            best_score = score
            best_split = (train_idx, test_idx, train_rate, test_rate, seed_start + offset)

    return best_split


def select_features(X_train, y_train, max_features=60):
    non_constant_columns = X_train.columns[X_train.nunique(dropna=False) > 1]
    dropped_columns = len(X_train.columns) - len(non_constant_columns)
    X_train = X_train[non_constant_columns]
    k = min(max_features, X_train.shape[1])
    print(f"\nSelecting up to {k} features using training data only...")
    if dropped_columns:
        print(f"Dropped {dropped_columns} constant training-only features before selection")

    selector_stats = SelectKBest(score_func=f_classif, k=k)
    selector_stats.fit(X_train, y_train)
    stats_features = list(X_train.columns[selector_stats.get_support()])

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    rf_top_features = list(rf_importance.head(k).index)

    rfe_estimator = RandomForestClassifier(
        n_estimators=150,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rfe = RFE(rfe_estimator, n_features_to_select=k)
    rfe.fit(X_train, y_train)
    rfe_features = list(X_train.columns[rfe.support_])

    vote_counts = {}
    for feature in set(stats_features) | set(rf_top_features) | set(rfe_features):
        vote_counts[feature] = int(feature in stats_features) + int(feature in rf_top_features) + int(feature in rfe_features)

    ranked_features = sorted(
        vote_counts,
        key=lambda feature: (
            vote_counts[feature],
            rf_importance.get(feature, 0.0),
            selector_stats.scores_[X_train.columns.get_loc(feature)],
        ),
        reverse=True,
    )
    selected_features = ranked_features[:k]

    print(f"Selected {len(selected_features)} features")
    print(f"Top selected features: {selected_features[:10]}")
    return selected_features, rf_importance[selected_features].sort_values(ascending=False)


class ImprovedLandslideANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def to_tensor_dataset(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    return X_tensor, y_tensor, TensorDataset(X_tensor, y_tensor)


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


def tune_threshold(y_true, probabilities):
    thresholds = np.linspace(0.20, 0.80, 61)
    scored = []
    for threshold in thresholds:
        metrics = compute_metrics(y_true, probabilities, float(threshold))
        scored.append(metrics)

    best = max(
        scored,
        key=lambda item: (
            item["mcc"],
            item["f1"],
            item["pr_auc"] if not np.isnan(item["pr_auc"]) else -np.inf,
        ),
    )
    return best, scored


def evaluate_model(model, X_tensor, y_tensor, threshold, split_name, save_plot_path=None):
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
    y_true = y_tensor.cpu().numpy().astype(int).flatten()
    metrics = compute_metrics(y_true, probabilities, threshold)

    print(f"\n=== {split_name.upper()} METRICS (threshold={threshold:.3f}) ===")
    print(f"{'Accuracy':<12} {metrics['accuracy']:.4f}")
    print(f"{'Precision':<12} {metrics['precision']:.4f}")
    print(f"{'Recall':<12} {metrics['recall']:.4f}")
    print(f"{'F1':<12} {metrics['f1']:.4f}")
    print(f"{'AUROC':<12} {metrics['auroc']:.4f}")
    print(f"{'PR-AUC':<12} {metrics['pr_auc']:.4f}")
    print(f"{'MCC':<12} {metrics['mcc']:.4f}")
    print(f"{'Brier':<12} {metrics['brier']:.4f}")

    predictions = (probabilities >= threshold).astype(int)
    print("\nClassification report:")
    print(classification_report(y_true, predictions, target_names=["Non-Landslide", "Landslide"]))

    if save_plot_path is not None and len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
        cm = confusion_matrix(y_true, predictions)

        plt.figure(figsize=(18, 4.5))

        plt.subplot(1, 4, 1)
        plt.plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.title(f"{split_name} ROC")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 2)
        plt.plot(recall_curve, precision_curve, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
        plt.title(f"{split_name} Precision-Recall")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 4, 3)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-Landslide", "Landslide"],
            yticklabels=["Non-Landslide", "Landslide"],
        )
        plt.title(f"{split_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.subplot(1, 4, 4)
        plt.hist(probabilities[y_true == 0], bins=30, alpha=0.7, label="Non-Landslide")
        plt.hist(probabilities[y_true == 1], bins=30, alpha=0.7, label="Landslide")
        plt.axvline(threshold, linestyle="--", color="black", label=f"Threshold {threshold:.2f}")
        plt.title(f"{split_name} Probability Distribution")
        plt.xlabel("Predicted probability")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    return metrics, probabilities, y_true


def save_training_curves(train_losses, val_losses, threshold_scores):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title("Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    thresholds = [item["threshold"] for item in threshold_scores]
    mcc_scores = [item["mcc"] for item in threshold_scores]
    f1_scores = [item["f1"] for item in threshold_scores]
    plt.plot(thresholds, mcc_scores, label="Validation MCC")
    plt.plot(thresholds, f1_scores, label="Validation F1")
    plt.title("Threshold Search")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(TRAINING_RESULTS_DIR / "training_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    landslide_csv, non_landslide_csv = resolve_csvs()
    print(f"Using landslide CSV: {landslide_csv}")
    print(f"Using non-landslide CSV: {non_landslide_csv}")

    landslides = pd.read_csv(landslide_csv)
    non_landslides = pd.read_csv(non_landslide_csv)
    landslides["label"] = 1
    non_landslides["label"] = 0

    full_data = pd.concat([landslides, non_landslides], ignore_index=True)
    full_data = full_data.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X = full_data.drop(columns=["label"]).replace({True: 1, False: 0})
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = full_data["label"].astype(int)
    coordinates = X[["xcoord", "ycoord"]].copy()
    X = X.drop(columns=["xcoord", "ycoord", "fid"], errors="ignore")

    print(f"Dataset shape: {X.shape}")
    print(f"Positive class rate: {y.mean():.3f}")
    print(
        f"Coordinate ranges: X({coordinates['xcoord'].min():.0f} to {coordinates['xcoord'].max():.0f}), "
        f"Y({coordinates['ycoord'].min():.0f} to {coordinates['ycoord'].max():.0f})"
    )

    block_ids = create_spatial_blocks(coordinates, n_blocks=N_BLOCKS)
    block_summary = pd.DataFrame({"block": block_ids, "label": y}).groupby("block")["label"].agg(["count", "mean"])
    valid_blocks = block_summary[block_summary["count"] >= MIN_BLOCK_SIZE].index.to_numpy()
    valid_mask = np.isin(block_ids, valid_blocks)

    X_spatial = X.loc[valid_mask].reset_index(drop=True)
    y_spatial = y.loc[valid_mask].reset_index(drop=True)
    coordinates_spatial = coordinates.loc[valid_mask].reset_index(drop=True)
    block_ids_spatial = block_ids[valid_mask]

    print(f"Keeping {len(X_spatial):,}/{len(X):,} samples in blocks with >= {MIN_BLOCK_SIZE} samples")
    print(f"Spatial dataset shape: {X_spatial.shape}")

    trainval_idx, test_idx, trainval_rate, test_rate, test_seed = find_balanced_group_split(
        X_spatial,
        y_spatial,
        block_ids_spatial,
        test_size=0.2,
    )

    X_trainval = X_spatial.iloc[trainval_idx].reset_index(drop=True)
    X_test = X_spatial.iloc[test_idx].reset_index(drop=True)
    y_trainval = y_spatial.iloc[trainval_idx].reset_index(drop=True)
    y_test = y_spatial.iloc[test_idx].reset_index(drop=True)
    trainval_blocks = block_ids_spatial[trainval_idx]

    train_idx, val_idx, train_rate, val_rate, val_seed = find_balanced_group_split(
        X_trainval,
        y_trainval,
        trainval_blocks,
        test_size=0.25,
    )

    X_train = X_trainval.iloc[train_idx].reset_index(drop=True)
    X_val = X_trainval.iloc[val_idx].reset_index(drop=True)
    y_train = y_trainval.iloc[train_idx].reset_index(drop=True)
    y_val = y_trainval.iloc[val_idx].reset_index(drop=True)
    train_blocks = trainval_blocks[train_idx]

    print("\n=== Spatial Split Summary ===")
    print(f"Balanced test split seed: {test_seed}")
    print(f"Balanced validation split seed: {val_seed}")
    print(f"Train:      {len(X_train):,} samples, landslide rate={y_train.mean():.3f}")
    print(f"Validation: {len(X_val):,} samples, landslide rate={y_val.mean():.3f}")
    print(f"Test:       {len(X_test):,} samples, landslide rate={y_test.mean():.3f}")

    selected_features, selected_feature_importance = select_features(X_train, y_train, max_features=MAX_SELECTED_FEATURES)
    X_train_sel = X_train[selected_features].copy()
    X_val_sel = X_val[selected_features].copy()
    X_test_sel = X_test[selected_features].copy()

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_features)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_sel), columns=selected_features)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_features)

    print("\nRobustScaler fitted on training split only")

    X_train_tensor, y_train_tensor, train_dataset = to_tensor_dataset(X_train_scaled.values, y_train)
    X_val_tensor, y_val_tensor, val_dataset = to_tensor_dataset(X_val_scaled.values, y_val)
    X_test_tensor, y_test_tensor, _ = to_tensor_dataset(X_test_scaled.values, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    pos_weight_value = float(class_weights[1] / class_weights[0])
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)

    print(f"Class weights: {dict(enumerate(class_weights))}")
    print(f"Positive class BCE weight: {pos_weight_value:.3f}")

    model = ImprovedLandslideANN(X_train_scaled.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n=== Training Improved ANN ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item()

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                running_val_loss += criterion(outputs, y_batch).item()

        avg_train_loss = running_train_loss / max(len(train_loader), 1)
        avg_val_loss = running_val_loss / max(len(val_loader), 1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | "
                f"train={avg_train_loss:.4f} | val={avg_val_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=False))
    print(f"\nLoaded best checkpoint from {BEST_MODEL_PATH}")

    model.eval()
    with torch.no_grad():
        val_probabilities = torch.sigmoid(model(X_val_tensor)).cpu().numpy().flatten()
    y_val_np = y_val_tensor.cpu().numpy().astype(int).flatten()
    best_threshold_metrics, threshold_scores = tune_threshold(y_val_np, val_probabilities)
    best_threshold = best_threshold_metrics["threshold"]

    print(
        f"\nBest validation threshold: {best_threshold:.3f} "
        f"(MCC={best_threshold_metrics['mcc']:.4f}, F1={best_threshold_metrics['f1']:.4f})"
    )

    save_training_curves(train_losses, val_losses, threshold_scores)

    val_metrics, _, _ = evaluate_model(
        model,
        X_val_tensor,
        y_val_tensor,
        threshold=best_threshold,
        split_name="Validation",
        save_plot_path=TRAINING_RESULTS_DIR / "advanced_validation_evaluation.png",
    )
    test_default_metrics, _, _ = evaluate_model(
        model,
        X_test_tensor,
        y_test_tensor,
        threshold=0.5,
        split_name="Test Default",
        save_plot_path=None,
    )
    test_optimized_metrics, _, _ = evaluate_model(
        model,
        X_test_tensor,
        y_test_tensor,
        threshold=best_threshold,
        split_name="Test Optimized",
        save_plot_path=SCRIPT_DIR / "advanced_evaluation.png",
    )

    final_metrics_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "AUROC", "PR-AUC", "MCC", "Brier", "Threshold"],
            "Validation": [
                val_metrics["accuracy"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                val_metrics["auroc"],
                val_metrics["pr_auc"],
                val_metrics["mcc"],
                val_metrics["brier"],
                val_metrics["threshold"],
            ],
            "Test_0.5": [
                test_default_metrics["accuracy"],
                test_default_metrics["precision"],
                test_default_metrics["recall"],
                test_default_metrics["f1"],
                test_default_metrics["auroc"],
                test_default_metrics["pr_auc"],
                test_default_metrics["mcc"],
                test_default_metrics["brier"],
                test_default_metrics["threshold"],
            ],
            "Test_Optimized": [
                test_optimized_metrics["accuracy"],
                test_optimized_metrics["precision"],
                test_optimized_metrics["recall"],
                test_optimized_metrics["f1"],
                test_optimized_metrics["auroc"],
                test_optimized_metrics["pr_auc"],
                test_optimized_metrics["mcc"],
                test_optimized_metrics["brier"],
                test_optimized_metrics["threshold"],
            ],
        }
    )
    final_metrics_df.to_csv(TRAINING_RESULTS_DIR / "final_test_metrics.csv", index=False)

    feature_importance_df = selected_feature_importance.reset_index()
    feature_importance_df.columns = ["feature", "importance"]
    feature_importance_df.to_csv(TRAINING_RESULTS_DIR / "selected_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 8))
    top_features = selected_feature_importance.head(15)
    plt.barh(top_features.index[::-1], top_features.values[::-1])
    plt.xlabel("Random Forest importance")
    plt.title("Top 15 Selected Features")
    plt.tight_layout()
    plt.savefig(TRAINING_RESULTS_DIR / "feature_importance_advanced.png", dpi=300, bbox_inches="tight")
    plt.close()

    metadata = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "selected_features": selected_features,
        "best_threshold": best_threshold,
        "model_architecture": "ImprovedLandslideANN",
        "feature_selection_method": "ensemble_training_only",
        "class_weights": {0: float(class_weights[0]), 1: float(class_weights[1])},
        "input_dim": int(X_train_scaled.shape[1]),
        "device": "cpu",
        "training_metrics": {
            "validation": val_metrics,
            "test_default": test_default_metrics,
            "test_optimized": test_optimized_metrics,
        },
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
    }
    torch.save(metadata, MODEL_PACKAGE_PATH)
    summary = {
        "model_path": str(MODEL_PACKAGE_PATH),
        "num_selected_features": len(selected_features),
        "best_threshold": best_threshold,
        "validation_metrics": val_metrics,
        "test_default_metrics": test_default_metrics,
        "test_optimized_metrics": test_optimized_metrics,
    }
    with open(TRAINING_RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Training Summary ===")
    print(f"Saved model package: {MODEL_PACKAGE_PATH}")
    print(f"Saved metrics CSV: {TRAINING_RESULTS_DIR / 'final_test_metrics.csv'}")
    print(f"Saved summary JSON: {TRAINING_RESULTS_DIR / 'training_summary.json'}")
    print(f"Selected features: {len(selected_features)}")
    print(f"Best deployment threshold: {best_threshold:.3f}")

    print("\nReminder: `train.py` still approximates the original MinMax scaling because the raw pre-normalization tables are not in this repo.")
    print("The new model should be cleaner and less leakage-prone, but the susceptibility map quality still depends on preprocessing consistency.")


if __name__ == "__main__":
    main()
