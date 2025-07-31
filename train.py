import rasterio
import numpy as np
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import gc

# Memory management settings
CHUNK_SIZE = 50000  # Process this many pixels at a time to avoid memory crashes

print("Loading raster files...")
# Define the expected raster order to match training data
expected_raster_names = [
    'aspect_utm15_aligned.tif',
    'elv_aligned.tif', 
    'flow_acc_aligned.tif',
    'planCurv_aligned.tif',
    'profCurv_aligned.tif',
    'riversprox_aligned.tif',
    'roadsprox_aligned.tif',
    'slope_aligned.tif',
    'SPI_aligned.tif',
    'TPI_aligned.tif',
    'TRI_aligned.tif',
    'TWI_aligned.tif',
    'lithology_aligned.tif',
    'soil_aligned.tif'
]

# Get all available raster files
available_rasters = glob.glob("/home/anees/Dropbox/geoProject/alignedRasters/*.tif")
available_names = [f.split('/')[-1] for f in available_rasters]

# Create ordered raster paths
raster_paths = []
for expected_name in expected_raster_names:
    matching_rasters = [f for f in available_rasters if f.endswith(expected_name)]
    if matching_rasters:
        raster_paths.append(matching_rasters[0])
        print(f"  Found: {expected_name}")
    else:
        print(f"  WARNING: Missing raster {expected_name}")

print(f"Using {len(raster_paths)} raster files in correct order")

# Read and stack - but be more memory efficient
print("Reading raster data...")
arrays = []
for i, path in enumerate(raster_paths):
    print(f"  Reading {path.split('/')[-1]}...")
    with rasterio.open(path) as src:
        arrays.append(src.read(1))  # Read first band

# Convert to 3D array: (bands, height, width)
stacked = np.stack(arrays, axis=0)
print("Shape of stacked raster:", stacked.shape)  # (bands, height, width)

# Get dimensions
bands, height, width = stacked.shape
total_pixels = height * width

print(f"Total pixels to process: {total_pixels:,}")
print(f"Processing in chunks of {CHUNK_SIZE:,} pixels to manage memory")

# Load training data to get the exact feature structure and encoding mappings
print("Loading training data to understand feature structure...")
landslides = pd.read_csv("output_landslides.csv")
nonLandslides = pd.read_csv("output_non_landslides.csv")

# Combine datasets
combined = pd.concat([landslides, nonLandslides], ignore_index=True)

# Get all feature columns (excluding metadata)
feature_cols = [col for col in combined.columns if col not in ['fid', 'xcoord', 'ycoord']]
print(f"Expected feature columns ({len(feature_cols)}): {feature_cols}")

# Get the exact order of columns used in training
columns_to_scale = ['aspect', 'elv', 'flowAcc', 'planCurv', 'profCurv',
                    'riverProx', 'roadProx', 'slope', 'SPI', 'TPI', 'TRI', 'TWI']

# Get one-hot encoded columns
lithology_cols = [col for col in feature_cols if col.startswith('lithology_')]
soil_cols = [col for col in feature_cols if col.startswith('soil_')]

print(f"Lithology columns ({len(lithology_cols)}): {lithology_cols}")
print(f"Soil columns ({len(soil_cols)}): {soil_cols}")

# Verify raster order matches expected features
columns_to_scale = ['aspect', 'elv', 'flowAcc', 'planCurv', 'profCurv',
                    'riverProx', 'roadProx', 'slope', 'SPI', 'TPI', 'TRI', 'TWI']

expected_raster_order = columns_to_scale + ['lithology', 'soil']
print(f"Expected feature order: {expected_raster_order}")

if len(raster_paths) != len(expected_raster_order):
    print(f"ERROR: Found {len(raster_paths)} rasters but expected {len(expected_raster_order)}")
    print("Cannot proceed with mismatched raster count")
    exit(1)

# Fit scaler on continuous variables only
scaler = MinMaxScaler()
scaler.fit(combined[columns_to_scale])

# Get unique values for one-hot encoding mapping
lithology_values = combined[lithology_cols].idxmax(axis=1).str.replace('lithology_', '').astype(int).unique()
soil_values = combined[soil_cols].idxmax(axis=1).str.replace('soil_', '').astype(int).unique()

print(f"Unique lithology values: {sorted(lithology_values)}")
print(f"Unique soil values: {sorted(soil_values)}")

# Load model
print("Loading trained model...")
try:
    # Try loading with weights_only=False (PyTorch 2.6+ compatibility)
    model_data = torch.load("landslide_model_advanced_complete.pth", weights_only=False)
    
    # Check if it's a dictionary containing the model
    if isinstance(model_data, dict):
        if 'model' in model_data:
            model = model_data['model']
        elif 'model_state_dict' in model_data:
            # Recreate the model architecture using saved info
            print("Recreating model from state dict...")
            
            # Get model architecture info
            input_dim = model_data['input_dim']
            print(f"Model input dimension: {input_dim}")
            
            # Recreate the model architecture (using the exact architecture from training)
            import torch.nn as nn
            import torch.nn.functional as F
            
            class AttentionLayer(nn.Module):
                def __init__(self, input_dim):
                    super(AttentionLayer, self).__init__()
                    self.attention = nn.Sequential(
                        nn.Linear(input_dim, input_dim // 2),
                        nn.ReLU(),
                        nn.Linear(input_dim // 2, input_dim),
                        nn.Softmax(dim=1)
                    )
                
                def forward(self, x):
                    attention_weights = self.attention(x)
                    return x * attention_weights

            class ResidualBlock(nn.Module):
                def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
                    super(ResidualBlock, self).__init__()
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
                    out += residual  # Residual connection
                    return self.relu(out)

            class AdvancedLandslideANN(nn.Module):
                def __init__(self, input_dim):
                    super(AdvancedLandslideANN, self).__init__()
                    self.input_layer = nn.Sequential(
                        nn.Linear(input_dim, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Dropout(0.4)
                    )
                    
                    # Attention mechanism
                    self.attention = AttentionLayer(512)
                    
                    # Residual blocks
                    self.res_block1 = ResidualBlock(512, 256, 0.3)
                    self.res_block2 = ResidualBlock(512, 256, 0.3)
                    
                    # Feature extraction layers
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
                        nn.Dropout(0.1)
                    )
                    
                    # Output layer
                    self.output = nn.Linear(64, 1)
                    
                def forward(self, x):
                    x = self.input_layer(x)
                    x = self.attention(x)
                    x = self.res_block1(x)
                    x = self.res_block2(x)
                    x = self.feature_layers(x)
                    return self.output(x)
            
            # Create model and load state dict
            model = AdvancedLandslideANN(input_dim)
            model.load_state_dict(model_data['model_state_dict'])
            
            # Don't use the saved scaler - it was fitted on one-hot encoded data
            # We'll use our own scaler fitted only on continuous features
            print("Using local scaler fitted on continuous features only")
            
            # Also load the saved threshold if available
            best_threshold = model_data.get('best_threshold', 0.5)
            print(f"Using threshold: {best_threshold}")
            
            # Get selected features if available (important for feature matching)
            if 'selected_features' in model_data:
                selected_features = model_data['selected_features']
                print(f"Model was trained on {len(selected_features)} selected features")
                print(f"First few selected features: {selected_features[:10]}")
            else:
                selected_features = None
        else:
            print("Model file structure not recognized. Available keys:", list(model_data.keys()))
            exit(1)
    else:
        model = model_data
        
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

model.eval()
print("Model loaded successfully!")
print(f"Expected input features: {input_dim}")

# Create output array for predictions
print("Initializing output arrays...")
full_prediction = np.full((height, width), np.nan, dtype=np.float32)

# Process data in chunks to avoid memory issues
print("Starting prediction in chunks...")
for chunk_start in range(0, total_pixels, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_pixels)
    chunk_size = chunk_end - chunk_start
    
    print(f"Processing chunk {chunk_start//CHUNK_SIZE + 1}/{(total_pixels-1)//CHUNK_SIZE + 1}: pixels {chunk_start:,} to {chunk_end-1:,}")
    
    # Extract chunk from stacked rasters
    chunk_data = stacked.reshape(bands, -1)[:, chunk_start:chunk_end].T  # Shape: (chunk_size, bands)
    
    # Check for valid pixels (no NaN values)
    valid_mask_chunk = ~np.isnan(chunk_data).any(axis=1)
    
    if not valid_mask_chunk.any():
        print("  No valid pixels in this chunk, skipping...")
        continue
    
    # Get valid data for this chunk
    valid_chunk_data = chunk_data[valid_mask_chunk]
    print(f"  Valid pixels in chunk: {valid_mask_chunk.sum():,}/{chunk_size:,}")
    
    # Check for edge artifacts - identify if we're processing edge pixels
    chunk_positions_all = np.arange(chunk_start, chunk_end)
    chunk_rows_all = chunk_positions_all // width
    chunk_cols_all = chunk_positions_all % width
    
    # Identify edge pixels (within 50 pixels of border)
    edge_buffer = 50
    is_edge_chunk = (
        (chunk_cols_all < edge_buffer).any() or 
        (chunk_cols_all >= width - edge_buffer).any() or
        (chunk_rows_all < edge_buffer).any() or 
        (chunk_rows_all >= height - edge_buffer).any()
    )
    
    if is_edge_chunk:
        print(f"    Note: Processing edge region (within {edge_buffer} pixels of border)")
    
    # Separate continuous and categorical data
    continuous_data = valid_chunk_data[:, :len(columns_to_scale)]  # First 12 columns
    lithology_raw = valid_chunk_data[:, len(columns_to_scale)]    # 13th column (lithology)
    soil_raw = valid_chunk_data[:, len(columns_to_scale)+1]       # 14th column (soil)
    
    # Create DataFrame for this chunk
    chunk_df = pd.DataFrame(continuous_data, columns=columns_to_scale)
    
    # Scale continuous variables
    continuous_scaled = scaler.transform(chunk_df)
    
    # One-hot encode lithology
    lithology_encoded = np.zeros((len(valid_chunk_data), len(lithology_cols)))
    for i, val in enumerate(lithology_raw):
        if not np.isnan(val):
            val_int = int(val)
            col_name = f'lithology_{val_int}'
            if col_name in lithology_cols:
                col_idx = lithology_cols.index(col_name)
                lithology_encoded[i, col_idx] = 1
    
    # One-hot encode soil
    soil_encoded = np.zeros((len(valid_chunk_data), len(soil_cols)))
    for i, val in enumerate(soil_raw):
        if not np.isnan(val):
            val_int = int(val)
            col_name = f'soil_{val_int}'
            if col_name in soil_cols:
                col_idx = soil_cols.index(col_name)
                soil_encoded[i, col_idx] = 1
    
    # Combine all features
    chunk_features = np.concatenate([continuous_scaled, lithology_encoded, soil_encoded], axis=1)
    
    # Apply feature selection if it was used during training
    if selected_features is not None:
        # Create feature names for current chunk
        feature_names = columns_to_scale + lithology_cols + soil_cols
        # Select only the features that were used during training
        feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
        chunk_features = chunk_features[:, feature_indices]
        print(f"  Applied feature selection: {chunk_features.shape[1]} features")
    
    # Convert to tensor and predict
    chunk_tensor = torch.tensor(chunk_features, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(chunk_tensor)
        # Apply sigmoid to get probabilities between 0 and 1 (susceptibility scores)
        susceptibility_scores = torch.sigmoid(predictions).cpu().numpy().flatten()
    
    # Apply edge correction during prediction for pixels near borders
    if is_edge_chunk:
        # Get positions for valid pixels in this chunk
        chunk_positions_valid = chunk_positions_all[valid_mask_chunk]
        chunk_rows_valid = chunk_positions_valid // width
        chunk_cols_valid = chunk_positions_valid % width
        
        # Identify which pixels are actually near edges
        edge_buffer = 50
        near_edge_mask = (
            (chunk_cols_valid < edge_buffer) |
            (chunk_cols_valid >= width - edge_buffer) |
            (chunk_rows_valid < edge_buffer) |
            (chunk_rows_valid >= height - edge_buffer)
        )
        
        if near_edge_mask.any():
            # Apply conservative smoothing to edge predictions
            edge_susceptibility = susceptibility_scores[near_edge_mask]
            
            # Cap extreme values at edges (reduce high predictions near borders)
            edge_cap = 0.7  # Maximum allowed susceptibility at edges
            edge_susceptibility = np.minimum(edge_susceptibility, edge_cap)
            
            # Apply distance-based dampening for pixels very close to edge
            for idx in np.where(near_edge_mask)[0]:
                row, col = chunk_rows_valid[idx], chunk_cols_valid[idx]
                dist_to_edge = min(col, row, width-1-col, height-1-row)
                
                if dist_to_edge < edge_buffer:
                    # Apply dampening factor based on distance to edge
                    dampen_factor = 0.5 + 0.5 * (dist_to_edge / edge_buffer)
                    edge_susceptibility[np.where(near_edge_mask)[0] == idx] *= dampen_factor
            
            susceptibility_scores[near_edge_mask] = edge_susceptibility
            print(f"    Applied edge correction to {near_edge_mask.sum()} border pixels")
    
    # Map predictions back to full raster positions
    chunk_positions = np.arange(chunk_start, chunk_end)[valid_mask_chunk]
    chunk_rows = chunk_positions // width
    chunk_cols = chunk_positions % width
    
    full_prediction[chunk_rows, chunk_cols] = susceptibility_scores
    
    # Clear memory
    del chunk_data, valid_chunk_data, chunk_df, continuous_scaled
    del lithology_encoded, soil_encoded, chunk_features, chunk_tensor
    del predictions, susceptibility_scores
    gc.collect()
    
    print(f"  Chunk completed. Memory cleared.")

print("Prediction completed!")

# Analyze edge effects and data distribution
print("Analyzing prediction distribution and edge effects...")
valid_predictions = full_prediction[~np.isnan(full_prediction)]
print(f"  Valid prediction statistics:")
print(f"    Min: {np.min(valid_predictions):.4f}")
print(f"    Max: {np.max(valid_predictions):.4f}")
print(f"    Mean: {np.mean(valid_predictions):.4f}")
print(f"    Median: {np.median(valid_predictions):.4f}")
print(f"    Std: {np.std(valid_predictions):.4f}")

# Check for edge artifacts by examining border regions
print("  Checking edge regions for artifacts...")
left_edge = full_prediction[:, :100]  # First 100 columns
right_edge = full_prediction[:, -100:]  # Last 100 columns
top_edge = full_prediction[:100, :]  # First 100 rows  
bottom_edge = full_prediction[-100:, :]  # Last 100 rows

def edge_stats(edge_data, edge_name):
    valid_edge = edge_data[~np.isnan(edge_data)]
    if len(valid_edge) > 0:
        print(f"    {edge_name}: mean={np.mean(valid_edge):.4f}, max={np.max(valid_edge):.4f}, high-risk%={((valid_edge >= best_threshold).sum()/len(valid_edge)*100):.1f}%")
    else:
        print(f"    {edge_name}: No valid data")

edge_stats(left_edge, "Left edge  ")
edge_stats(right_edge, "Right edge ")
edge_stats(top_edge, "Top edge   ")
edge_stats(bottom_edge, "Bottom edge")

# Check for abrupt transitions (high gradient areas)
print("  Checking for abrupt transitions...")
# Calculate gradients to find sudden changes
from scipy import ndimage
grad_x = ndimage.sobel(full_prediction, axis=1)  # Horizontal gradient
grad_y = ndimage.sobel(full_prediction, axis=0)  # Vertical gradient
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Find areas with very high gradients (abrupt changes)
high_gradient_threshold = np.nanpercentile(gradient_magnitude, 99)  # Top 1% of gradients
high_gradient_mask = gradient_magnitude > high_gradient_threshold

print(f"    High gradient threshold: {high_gradient_threshold:.4f}")
print(f"    Pixels with abrupt transitions: {np.sum(high_gradient_mask):,}")

# Save gradient magnitude for inspection
with rasterio.open(raster_paths[0]) as src:
    meta_grad = src.meta.copy()

meta_grad.update({
    "count": 1,
    "dtype": 'float32'
})

with rasterio.open("gradient_magnitude.tif", "w", **meta_grad) as dst:
    dst.write(gradient_magnitude.astype(np.float32), 1)

print("    Gradient magnitude saved as 'gradient_magnitude.tif'")

# Also save a masked version showing only valid predictions for easier inspection
masked_prediction = full_prediction.copy()
masked_prediction[np.isnan(full_prediction)] = -1  # Set NoData to -1 for visualization

with rasterio.open("susceptibility_map_masked.tif", "w", **meta_grad) as dst:
    dst.write(masked_prediction, 1)

print("    Masked susceptibility map saved as 'susceptibility_map_masked.tif'")

# Save the susceptibility map
print("Saving susceptibility map...")
# Get metadata from first raster
with rasterio.open(raster_paths[0]) as src:
    meta = src.meta.copy()

meta.update({
    "count": 1,
    "dtype": 'float32'
})

# Save the susceptibility map
with rasterio.open("susceptibility_map.tif", "w", **meta) as dst:
    dst.write(full_prediction, 1)

print("Susceptibility map saved as 'susceptibility_map.tif'")
print(f"Prediction statistics:")
print(f"  Total pixels: {height * width:,}")
print(f"  Valid predictions: {(~np.isnan(full_prediction)).sum():,}")
print(f"  Susceptibility range: {np.nanmin(full_prediction):.3f} to {np.nanmax(full_prediction):.3f}")
print(f"  Mean susceptibility: {np.nanmean(full_prediction):.3f}")
print(f"  High-risk pixels (>= {best_threshold:.3f}): {(full_prediction >= best_threshold).sum():,}")
print(f"  Low-risk pixels (< {best_threshold:.3f}): {(full_prediction < best_threshold).sum():,}")
print(f"  Percentage high-risk: {((full_prediction >= best_threshold).sum() / (~np.isnan(full_prediction)).sum() * 100):.2f}%")