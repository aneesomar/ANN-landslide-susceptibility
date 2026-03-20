import gc
import glob
import math
import os
from contextlib import ExitStack

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from project_paths import (
    SUSCEPTIBILITY_MAPS_DIR,
    ensure_project_dirs,
    resolve_processed_csvs,
    resolve_processed_landslide_csv,
)


WINDOW_SIZE = 512
EDGE_BUFFER = 50
EXPECTED_RASTER_NAMES = [
    "aspect_utm15_aligned.tif",
    "elv_aligned.tif",
    "flow_acc_aligned.tif",
    "planCurv_aligned.tif",
    "profCurv_aligned.tif",
    "riversprox_aligned.tif",
    "roadsprox_aligned.tif",
    "slope_aligned.tif",
    "SPI_aligned.tif",
    "TPI_aligned.tif",
    "TRI_aligned.tif",
    "TWI_aligned.tif",
    "lithology_aligned.tif",
    "soil_aligned.tif",
]
CONTINUOUS_COLUMNS = [
    "aspect",
    "elv",
    "flowAcc",
    "planCurv",
    "profCurv",
    "riverProx",
    "roadProx",
    "slope",
    "SPI",
    "TPI",
    "TRI",
    "TWI",
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ensure_project_dirs()


def find_raster_paths():
    print("Loading raster files...")
    possible_paths = [
        os.path.join(SCRIPT_DIR, "alignedRaster", "*.tif"),
        os.path.join(SCRIPT_DIR, "alignedRasters", "*.tif"),
        os.path.join(SCRIPT_DIR, "..", "alignedRaster", "*.tif"),
        os.path.join(SCRIPT_DIR, "..", "alignedRasters", "*.tif"),
        os.path.join(SCRIPT_DIR, "..", "..", "alignedRasters", "*.tif"),
        os.path.join(SCRIPT_DIR, "..", "..", "OneDrive", "geoProject", "alignedRasters", "*.tif"),
        os.path.join(SCRIPT_DIR, "..", "..", "..", "OneDrive", "geoProject", "alignedRasters", "*.tif"),
    ]

    available_rasters = []
    for path_pattern in possible_paths:
        found = glob.glob(path_pattern)
        if found:
            available_rasters = found
            print(f"  Found rasters at: {path_pattern}")
            break

    if not available_rasters:
        raise FileNotFoundError(
            f"Could not find aligned rasters. Searched: {possible_paths}"
        )

    raster_paths = []
    for expected_name in EXPECTED_RASTER_NAMES:
        matching_rasters = [path for path in available_rasters if path.endswith(expected_name)]
        if not matching_rasters:
            raise FileNotFoundError(f"Missing required raster: {expected_name}")
        raster_paths.append(matching_rasters[0])
        print(f"  Found: {expected_name}")

    print(f"Using {len(raster_paths)} raster files in correct order")
    return raster_paths


def clean_raster_data(data, nodata_value):
    clean = data.astype(np.float32, copy=False)
    if nodata_value is not None and not np.isnan(nodata_value):
        clean = np.where(clean == nodata_value, np.nan, clean)
    clean = np.where(clean <= -99999, np.nan, clean)
    clean = np.where(clean == -9999, np.nan, clean)
    clean = np.where(np.abs(clean) > 1e10, np.nan, clean)
    return clean


def iter_windows(width, height, window_size):
    for row_off in range(0, height, window_size):
        for col_off in range(0, width, window_size):
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=min(window_size, width - col_off),
                height=min(window_size, height - row_off),
            )


def resolve_training_csvs():
    print("Loading training data to understand feature structure...")
    landslides_path, non_landslides_path = resolve_processed_csvs()
    print(f"  Found CSV files at: {landslides_path}")
    landslides = pd.read_csv(landslides_path)
    non_landslides = pd.read_csv(non_landslides_path)
    return pd.concat([landslides, non_landslides], ignore_index=True)


def resolve_landslide_csv_path():
    return str(resolve_processed_landslide_csv())


def recover_training_minmax(landslides_df, raster_paths):
    print("\nRecovering original MinMax scaling from landslide training points...")
    coords = list(zip(landslides_df["xcoord"].to_numpy(), landslides_df["ycoord"].to_numpy()))
    recovered_mins = []
    recovered_maxs = []

    for column_name, raster_path in zip(CONTINUOUS_COLUMNS, raster_paths[: len(CONTINUOUS_COLUMNS)]):
        with rasterio.open(raster_path) as src:
            print(f"  Sampling {os.path.basename(raster_path)} for {column_name}...")
            sampled = np.array([value[0] for value in src.sample(coords)], dtype=np.float32)

        sampled = clean_raster_data(sampled, None)
        normalized_values = pd.to_numeric(landslides_df[column_name], errors="coerce").to_numpy(dtype=np.float32)
        valid_mask = ~np.isnan(sampled) & ~np.isnan(normalized_values)

        if not valid_mask.any():
            raise RuntimeError(
                f"Could not recover scaler for {column_name}: no valid raster/training pairs"
            )

        raw_values = sampled[valid_mask]
        min_value = float(raw_values.min())
        max_value = float(raw_values.max())

        if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
            raise RuntimeError(
                f"Could not recover scaler for {column_name}: invalid range {min_value}..{max_value}"
            )

        recovered_mins.append(min_value)
        recovered_maxs.append(max_value)
        print(f"    Recovered landslide-fit range: {min_value:.2f} to {max_value:.2f}")

    return np.array(recovered_mins, dtype=np.float32), np.array(recovered_maxs, dtype=np.float32)


def apply_recovered_minmax(values, mins, maxs):
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    return (values - mins) / denom


def load_model():
    print("Loading trained model...")
    model_paths = [
        os.path.join(SCRIPT_DIR, "models", "landslide_model_advanced_complete.pth"),
        os.path.join(SCRIPT_DIR, "landslide_model_advanced_complete.pth"),
        os.path.join(os.getcwd(), "landslide_model_advanced_complete.pth"),
        os.path.join(os.getcwd(), "models", "landslide_model_advanced_complete.pth"),
        os.path.join(SCRIPT_DIR, "..", "models", "landslide_model_advanced_complete.pth"),
    ]

    model_file = next((path for path in model_paths if os.path.exists(path)), None)
    if model_file is None:
        raise FileNotFoundError(
            f"Could not find landslide_model_advanced_complete.pth. Searched: {model_paths}"
        )

    print(f"  Found model at: {model_file}")
    model_data = torch.load(model_file, weights_only=False)

    robust_scaler = None
    selected_features = None
    best_threshold = 0.5
    input_dim = None

    if isinstance(model_data, dict):
        if "model" in model_data:
            model = model_data["model"]
            input_dim = getattr(getattr(model, "input_layer", None), "in_features", None)
        elif "model_state_dict" in model_data:
            import torch.nn as nn

            print("Recreating model from state dict...")
            input_dim = model_data["input_dim"]
            print(f"Model input dimension: {input_dim}")
            model_architecture = model_data.get("model_architecture", "AdvancedLandslideANN")

            class AttentionLayer(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.attention = nn.Sequential(
                        nn.Linear(input_dim, input_dim // 2),
                        nn.ReLU(),
                        nn.Linear(input_dim // 2, input_dim),
                        nn.Softmax(dim=1),
                    )

                def forward(self, x):
                    attention_weights = self.attention(x)
                    return x * attention_weights

            class ResidualBlock(nn.Module):
                def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.bn1 = nn.BatchNorm1d(hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, input_dim)
                    self.bn2 = nn.BatchNorm1d(input_dim)
                    self.dropout = nn.Dropout(dropout_rate)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    residual = x
                    out = self.relu(self.bn1(self.fc1(x)))
                    out = self.dropout(out)
                    out = self.bn2(self.fc2(out))
                    out += residual
                    return self.relu(out)

            class AdvancedLandslideANN(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.input_layer = nn.Sequential(
                        nn.Linear(input_dim, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                    )
                    self.attention = AttentionLayer(512)
                    self.res_block1 = ResidualBlock(512, 256, 0.3)
                    self.res_block2 = ResidualBlock(512, 256, 0.3)
                    self.feature_layers = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                    )
                    self.output = nn.Linear(64, 1)

                def forward(self, x):
                    x = self.input_layer(x)
                    x = self.attention(x)
                    x = self.res_block1(x)
                    x = self.res_block2(x)
                    x = self.feature_layers(x)
                    return self.output(x)

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

            if model_architecture == "ImprovedLandslideANN":
                model = ImprovedLandslideANN(input_dim)
            else:
                model = AdvancedLandslideANN(input_dim)
            model.load_state_dict(model_data["model_state_dict"])
            robust_scaler = model_data.get("scaler")
            best_threshold = model_data.get("best_threshold", 0.5)
            selected_features = model_data.get("selected_features")
        else:
            raise ValueError(
                f"Model file structure not recognized. Keys: {list(model_data.keys())}"
            )
    else:
        model = model_data

    model.eval()
    return model, robust_scaler, selected_features, best_threshold, input_dim


def apply_edge_correction(scores, rows, cols, width, height):
    near_edge_mask = (
        (cols < EDGE_BUFFER)
        | (cols >= width - EDGE_BUFFER)
        | (rows < EDGE_BUFFER)
        | (rows >= height - EDGE_BUFFER)
    )

    if not near_edge_mask.any():
        return scores

    adjusted_scores = scores.copy()
    edge_scores = np.minimum(adjusted_scores[near_edge_mask], 0.7)
    edge_rows = rows[near_edge_mask]
    edge_cols = cols[near_edge_mask]
    distances = np.minimum.reduce(
        [edge_cols, edge_rows, width - 1 - edge_cols, height - 1 - edge_rows]
    )
    dampening = 0.5 + 0.5 * (distances / EDGE_BUFFER)
    adjusted_scores[near_edge_mask] = edge_scores * dampening
    return adjusted_scores


def main():
    raster_paths = find_raster_paths()

    with rasterio.open(raster_paths[0]) as src:
        height = src.height
        width = src.width
        output_meta = src.meta.copy()

    for raster_path in raster_paths[1:]:
        with rasterio.open(raster_path) as src:
            if src.height != height or src.width != width:
                raise ValueError(
                    f"Raster shape mismatch for {raster_path}: "
                    f"expected {width}x{height}, got {src.width}x{src.height}"
                )

    total_pixels = height * width
    total_windows = math.ceil(height / WINDOW_SIZE) * math.ceil(width / WINDOW_SIZE)
    print(f"Total pixels to process: {total_pixels:,}")
    print(f"Processing in {total_windows:,} windows of up to {WINDOW_SIZE}x{WINDOW_SIZE}")

    combined = resolve_training_csvs()
    feature_cols = [col for col in combined.columns if col not in ["fid", "xcoord", "ycoord"]]
    lithology_cols = [col for col in feature_cols if col.startswith("lithology_")]
    soil_cols = [col for col in feature_cols if col.startswith("soil_")]

    print(f"Expected feature columns ({len(feature_cols)} total)")
    print(f"Lithology columns ({len(lithology_cols)}): {lithology_cols}")
    print(f"Soil columns ({len(soil_cols)}): {soil_cols}")

    expected_raster_order = CONTINUOUS_COLUMNS + ["lithology", "soil"]
    if len(raster_paths) != len(expected_raster_order):
        raise ValueError(
            f"Found {len(raster_paths)} rasters but expected {len(expected_raster_order)}"
        )

    landslide_csv_path = resolve_landslide_csv_path()
    landslides_only = pd.read_csv(landslide_csv_path)
    raster_mins, raster_maxs = recover_training_minmax(landslides_only, raster_paths)

    model, robust_scaler, selected_features, best_threshold, input_dim = load_model()
    if robust_scaler is not None:
        print("Loaded RobustScaler from training")
    else:
        print("WARNING: No RobustScaler found in model file")

    feature_names = CONTINUOUS_COLUMNS + lithology_cols + soil_cols
    feature_name_to_index = {name: index for index, name in enumerate(feature_names)}
    if selected_features is not None:
        feature_indices = [
            feature_name_to_index[name]
            for name in selected_features
            if name in feature_name_to_index
        ]
    else:
        feature_indices = list(range(len(feature_names)))
    selected_feature_names = [feature_names[i] for i in feature_indices]

    expected_input_dim = input_dim if input_dim is not None else len(feature_indices)
    if len(feature_indices) != expected_input_dim:
        raise ValueError(
            f"Feature count mismatch: model expects {expected_input_dim}, "
            f"but prediction pipeline produced {len(feature_indices)} features"
        )

    lithology_index = {name: index for index, name in enumerate(lithology_cols)}
    soil_index = {name: index for index, name in enumerate(soil_cols)}

    output_meta.update(count=1, dtype="float32", nodata=np.nan)
    output_path = str(SUSCEPTIBILITY_MAPS_DIR / "susceptibility_map.tif")

    print("\nStarting streamed prediction...")
    valid_prediction_count = 0
    high_risk_count = 0
    low_risk_count = 0
    running_sum = 0.0
    prediction_min = np.inf
    prediction_max = -np.inf

    with ExitStack() as stack:
        raster_sources = [stack.enter_context(rasterio.open(path)) for path in raster_paths]
        output_dst = stack.enter_context(rasterio.open(output_path, "w", **output_meta))

        for window_index, window in enumerate(iter_windows(width, height, WINDOW_SIZE), start=1):
            if window_index == 1 or window_index % 25 == 0 or window_index == total_windows:
                print(f"  Window {window_index:,}/{total_windows:,}")

            window_arrays = [
                clean_raster_data(src.read(1, window=window), src.nodata) for src in raster_sources
            ]
            window_stack = np.stack(window_arrays, axis=-1)
            pixel_matrix = window_stack.reshape(-1, len(raster_paths))

            valid_mask = ~np.isnan(pixel_matrix).any(axis=1)
            valid_mask &= ~(pixel_matrix == 0).all(axis=1)

            prediction_window = np.full(
                (int(window.height), int(window.width)),
                np.nan,
                dtype=np.float32,
            )

            if valid_mask.any():
                valid_pixels = pixel_matrix[valid_mask]
                continuous_scaled = apply_recovered_minmax(
                    valid_pixels[:, : len(CONTINUOUS_COLUMNS)],
                    raster_mins,
                    raster_maxs,
                )

                lithology_raw = valid_pixels[:, len(CONTINUOUS_COLUMNS)]
                soil_raw = valid_pixels[:, len(CONTINUOUS_COLUMNS) + 1]

                lithology_encoded = np.zeros((len(valid_pixels), len(lithology_cols)), dtype=np.float32)
                soil_encoded = np.zeros((len(valid_pixels), len(soil_cols)), dtype=np.float32)

                for row_index, value in enumerate(lithology_raw):
                    if not np.isnan(value):
                        col_name = f"lithology_{int(value)}"
                        if col_name in lithology_index:
                            lithology_encoded[row_index, lithology_index[col_name]] = 1.0

                for row_index, value in enumerate(soil_raw):
                    if not np.isnan(value):
                        col_name = f"soil_{int(value)}"
                        if col_name in soil_index:
                            soil_encoded[row_index, soil_index[col_name]] = 1.0

                chunk_features = np.concatenate(
                    [continuous_scaled, lithology_encoded, soil_encoded],
                    axis=1,
                )
                chunk_features = chunk_features[:, feature_indices]
                if robust_scaler is not None:
                    chunk_features = robust_scaler.transform(
                        pd.DataFrame(chunk_features, columns=selected_feature_names)
                    )

                chunk_tensor = torch.tensor(chunk_features, dtype=torch.float32)
                with torch.no_grad():
                    scores = torch.sigmoid(model(chunk_tensor)).cpu().numpy().flatten()

                local_rows, local_cols = np.divmod(
                    np.flatnonzero(valid_mask),
                    int(window.width),
                )
                global_rows = local_rows + int(window.row_off)
                global_cols = local_cols + int(window.col_off)
                scores = apply_edge_correction(scores, global_rows, global_cols, width, height)

                flat_prediction = prediction_window.reshape(-1)
                flat_prediction[valid_mask] = scores.astype(np.float32)

                valid_prediction_count += scores.size
                high_risk_count += int((scores >= best_threshold).sum())
                low_risk_count += int((scores < best_threshold).sum())
                running_sum += float(scores.sum())
                prediction_min = min(prediction_min, float(scores.min()))
                prediction_max = max(prediction_max, float(scores.max()))

                del valid_pixels, continuous_scaled
                del lithology_encoded, soil_encoded, chunk_features, chunk_tensor, scores

            output_dst.write(prediction_window, 1, window=window)

            del window_arrays, window_stack, pixel_matrix, prediction_window
            gc.collect()

    if valid_prediction_count == 0:
        raise RuntimeError("No valid predictions were produced. Check raster NoData values.")

    mean_prediction = running_sum / valid_prediction_count
    print("\nPrediction completed successfully")
    print(f"Susceptibility map saved as: {output_path}")
    print("Prediction statistics:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Valid predictions: {valid_prediction_count:,}")
    print(f"  Susceptibility range: {prediction_min:.3f} to {prediction_max:.3f}")
    print(f"  Mean susceptibility: {mean_prediction:.3f}")
    print(f"  High-risk pixels (>= {best_threshold:.3f}): {high_risk_count:,}")
    print(f"  Low-risk pixels (< {best_threshold:.3f}): {low_risk_count:,}")
    print(
        f"  Percentage high-risk: "
        f"{(high_risk_count / valid_prediction_count * 100):.2f}%"
    )


if __name__ == "__main__":
    main()
