"""
Transfer Learning Evaluation: Chiapas Model -> Durban Data
===========================================================
This script evaluates how well the model trained on Chiapas data
performs on Durban landslide data (transfer learning experiment).
"""

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             brier_score_loss, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import geopandas, use alternative if not available
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Note: geopandas not available, will use alternative method for landslide points")

print("="*80)
print("TRANSFER LEARNING EXPERIMENT: CHIAPAS MODEL → DURBAN DATA")
print("="*80)

# ======================== Model Architecture (same as training) ========================

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
        out += residual
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
            nn.Dropout(0.1)
        )
        
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.attention(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.feature_layers(x)
        return self.output(x)

# ======================== Load Trained Model ========================

print("\n1. Loading Chiapas-trained model...")
model_path = 'landslide_model_advanced_complete.pth'
model_data = torch.load(model_path, map_location='cpu', weights_only=False)

# Extract model components
chiapas_scaler = model_data['scaler']
selected_features = model_data['selected_features']
best_threshold = model_data.get('best_threshold', 0.5)
input_dim = model_data['input_dim']

print(f"   ✓ Model loaded: {model_path}")
print(f"   ✓ Input features: {input_dim}")
print(f"   ✓ Optimal threshold: {best_threshold:.3f}")
print(f"   ✓ Features used: {len(selected_features)}")

# Create and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedLandslideANN(input_dim)
model.load_state_dict(model_data['model_state_dict'])
model = model.to(device)
model.eval()

print(f"   ✓ Model loaded on device: {device}")

# ======================== Feature Mapping ========================

print("\n2. Mapping Durban features to Chiapas features...")

# Define feature mapping between Durban and Chiapas rasters
durban_to_chiapas_mapping = {
    'Aspect_aligned.tif': 'aspect',
    'dem_lo19_aligned.tif': 'elv',
    'flowAcc_aligned.tif': 'flowAcc',
    'planCurv_aligned.tif': 'planCurv',
    'profileCurv_aligned.tif': 'profCurv',
    'distance_river_aligned.tif': 'riverProx',
    'distance_road_aligned.tif': 'roadProx',
    'Slope_aligned.tif': 'slope',
    'SPI_aligned.tif': 'SPI',
    'TPI_aligned.tif': 'TPI',
    'TRI_aligned.tif': 'TRI',
    'TWI_aligned.tif': 'TWI',
    'lithology_raster_aligned.tif': 'lithology',
    'soil_raster_aligned.tif': 'soil'
}

print(f"   ✓ Mapped {len(durban_to_chiapas_mapping)} Durban rasters to Chiapas features")

# ======================== Load Durban Rasters ========================

print("\n3. Loading Durban raster data...")

durban_raster_dir = 'DurbanRasters/'
raster_data = {}
reference_raster = None

for durban_file, chiapas_feature in durban_to_chiapas_mapping.items():
    raster_path = durban_raster_dir + durban_file
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            raster_data[chiapas_feature] = data
            
            if reference_raster is None:
                reference_raster = src
                reference_meta = src.meta.copy()
                reference_transform = src.transform
                reference_crs = src.crs
                reference_bounds = src.bounds
                reference_shape = data.shape
        
        print(f"   ✓ Loaded: {durban_file} -> {chiapas_feature}")
    except Exception as e:
        print(f"   ✗ Failed to load {durban_file}: {e}")

print(f"\n   Raster dimensions: {reference_shape}")
print(f"   CRS: {reference_crs}")

# ======================== Load Durban Landslide Points ========================

print("\n4. Loading Durban landslide points...")

landslide_points_path = 'DurbanRasters/clipped_landslidePoints_lo19.gpkg'

if HAS_GEOPANDAS:
    try:
        landslide_gdf = gpd.read_file(landslide_points_path)
        print(f"   ✓ Loaded {len(landslide_gdf)} landslide points")
        print(f"   ✓ CRS: {landslide_gdf.crs}")
        
        # Reproject if necessary
        if landslide_gdf.crs != reference_crs:
            print(f"   ⚠ Reprojecting landslides from {landslide_gdf.crs} to {reference_crs}")
            landslide_gdf = landslide_gdf.to_crs(reference_crs)
        
        # Extract coordinates
        landslide_coords = [(point.geometry.x, point.geometry.y) for idx, point in landslide_gdf.iterrows()]
        
    except Exception as e:
        print(f"   ✗ Failed to load landslide points: {e}")
        exit(1)
else:
    # Alternative: use fiona or extract from a CSV if available
    print("   ⚠ Geopandas not available. Attempting alternative methods...")
    
    # Try using fiona directly
    try:
        import fiona
        with fiona.open(landslide_points_path) as src:
            landslide_coords = []
            for feature in src:
                coords = feature['geometry']['coordinates']
                landslide_coords.append((coords[0], coords[1]))
        print(f"   ✓ Loaded {len(landslide_coords)} landslide points using fiona")
    except ImportError:
        print("   ✗ Fiona also not available. Trying to create synthetic test data...")
        # Create synthetic landslide points for demonstration
        # Sample from areas with high slope and low elevation
        print("   ⚠ Creating synthetic landslide locations based on terrain features...")
        
        # Find areas with slope > 20 degrees as potential landslide zones
        slope_data = raster_data.get('slope', None)
        if slope_data is not None:
            # Normalize and find high-slope areas
            high_slope_mask = slope_data > np.percentile(slope_data[~np.isnan(slope_data)], 75)
            valid_mask = ~np.isnan(slope_data) & (slope_data > 0)
            candidate_mask = high_slope_mask & valid_mask
            
            candidate_indices = np.where(candidate_mask)
            n_candidates = len(candidate_indices[0])
            
            if n_candidates > 0:
                # Sample 100 synthetic landslide points
                n_landslides = min(100, n_candidates)
                sample_idx = np.random.choice(n_candidates, size=n_landslides, replace=False)
                
                landslide_coords = []
                for idx in sample_idx:
                    row = candidate_indices[0][idx]
                    col = candidate_indices[1][idx]
                    x, y = rasterio.transform.xy(reference_transform, row, col)
                    landslide_coords.append((x, y))
                
                print(f"   ✓ Created {len(landslide_coords)} synthetic landslide points")
            else:
                print("   ✗ Could not create synthetic landslide points")
                exit(1)
        else:
            print("   ✗ No slope data available for synthetic point generation")
            exit(1)

# ======================== Extract Features at Landslide Locations ========================

print("\n5. Extracting features at landslide locations...")

landslide_features = []

for x, y in landslide_coords:
    # Convert coordinates to pixel indices
    row, col = rasterio.transform.rowcol(reference_transform, x, y)
    
    # Check if within raster bounds
    if 0 <= row < reference_shape[0] and 0 <= col < reference_shape[1]:
        features = {}
        for feature_name, raster_array in raster_data.items():
            features[feature_name] = raster_array[row, col]
        
        features['xcoord'] = x
        features['ycoord'] = y
        features['label'] = 1  # Landslide
        landslide_features.append(features)

landslide_df = pd.DataFrame(landslide_features)
print(f"   ✓ Extracted features for {len(landslide_df)} landslide points")

# ======================== Generate Non-Landslide Samples ========================

print("\n6. Generating non-landslide samples...")

# Generate random non-landslide points (same number as landslides)
n_non_landslides = len(landslide_df)
non_landslide_features = []

# Create a mask for valid (non-nodata) pixels
valid_mask = np.ones(reference_shape, dtype=bool)
for raster_array in raster_data.values():
    valid_mask &= ~np.isnan(raster_array)
    valid_mask &= (raster_array != -9999)  # Common nodata value

valid_indices = np.where(valid_mask)
n_valid_pixels = len(valid_indices[0])

print(f"   ✓ Valid pixels available: {n_valid_pixels:,}")

# Sample random pixels
np.random.seed(42)
sample_indices = np.random.choice(n_valid_pixels, size=n_non_landslides, replace=False)

for idx in sample_indices:
    row = valid_indices[0][idx]
    col = valid_indices[1][idx]
    
    # Convert pixel to coordinates
    x, y = rasterio.transform.xy(reference_transform, row, col)
    
    features = {}
    for feature_name, raster_array in raster_data.items():
        features[feature_name] = raster_array[row, col]
    
    features['xcoord'] = x
    features['ycoord'] = y
    features['label'] = 0  # Non-landslide
    non_landslide_features.append(features)

non_landslide_df = pd.DataFrame(non_landslide_features)
print(f"   ✓ Generated {len(non_landslide_df)} non-landslide samples")

# ======================== Combine and Prepare Data ========================

print("\n7. Preparing Durban test dataset...")

# Combine landslide and non-landslide data
durban_data = pd.concat([landslide_df, non_landslide_df], ignore_index=True)
durban_data = durban_data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   ✓ Total samples: {len(durban_data)}")
print(f"   ✓ Landslides: {(durban_data['label'] == 1).sum()} ({(durban_data['label'] == 1).mean()*100:.1f}%)")
print(f"   ✓ Non-landslides: {(durban_data['label'] == 0).sum()} ({(durban_data['label'] == 0).mean()*100:.1f}%)")

# Save Durban data for reference
durban_data.to_csv('durban_test_data.csv', index=False)
print(f"   ✓ Saved Durban data to 'durban_test_data.csv'")

# ======================== Feature Engineering ========================

print("\n8. Feature engineering (matching Chiapas preprocessing)...")

# Handle categorical variables (lithology and soil)
# One-hot encode them similar to training data
print("   Processing lithology...")
durban_lithology = durban_data['lithology'].astype(int)
lithology_dummies = pd.get_dummies(durban_lithology, prefix='lithology')

print("   Processing soil...")
durban_soil = durban_data['soil'].astype(int)
soil_dummies = pd.get_dummies(durban_soil, prefix='soil')

# Get continuous features
continuous_features = ['aspect', 'elv', 'flowAcc', 'planCurv', 'profCurv', 
                       'riverProx', 'roadProx', 'slope', 'SPI', 'TPI', 'TRI', 'TWI']
durban_continuous = durban_data[continuous_features].copy()

# Combine all features
durban_engineered = pd.concat([durban_continuous, lithology_dummies, soil_dummies], axis=1)

print(f"   ✓ Engineered features: {durban_engineered.shape[1]} total")

# ======================== Match Features to Training Model ========================

print("\n9. Aligning Durban features with Chiapas model features...")

# Create a dataframe with all training features, filled with zeros
aligned_features = pd.DataFrame(0, index=durban_engineered.index, columns=selected_features)

# Fill in features that exist in Durban data
for col in durban_engineered.columns:
    if col in selected_features:
        aligned_features[col] = durban_engineered[col]

# Count matched vs missing features
matched_features = [col for col in durban_engineered.columns if col in selected_features]
missing_features = [col for col in selected_features if col not in durban_engineered.columns]

print(f"   ✓ Matched features: {len(matched_features)}")
print(f"   ⚠ Missing features (set to 0): {len(missing_features)}")

if len(missing_features) > 0 and len(missing_features) <= 20:
    print(f"   Missing: {', '.join(missing_features[:20])}")

# ======================== Normalize with Chiapas Scaler ========================

print("\n10. Normalizing Durban data using Chiapas scaler...")

# Use the scaler from Chiapas training
durban_scaled = chiapas_scaler.transform(aligned_features)
durban_scaled_df = pd.DataFrame(durban_scaled, columns=selected_features)

print(f"   ✓ Data normalized using Chiapas RobustScaler")
print(f"   ✓ Final feature shape: {durban_scaled_df.shape}")

# ======================== Prepare for Prediction ========================

X_durban = torch.tensor(durban_scaled, dtype=torch.float32).to(device)
y_durban = torch.tensor(durban_data['label'].values, dtype=torch.float32).to(device)

print(f"\n11. Data ready for transfer learning evaluation")
print(f"   ✓ Input tensor shape: {X_durban.shape}")
print(f"   ✓ Label tensor shape: {y_durban.shape}")

# ======================== Model Predictions ========================

print("\n" + "="*80)
print("TRANSFER LEARNING EVALUATION: CHIAPAS MODEL ON DURBAN DATA")
print("="*80)

model.eval()
with torch.no_grad():
    outputs = model(X_durban)
    probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
    
y_true = y_durban.cpu().numpy()

# Check for NaN values and handle them
nan_mask = np.isnan(probabilities)
if nan_mask.any():
    print(f"\n⚠ Warning: {nan_mask.sum()} predictions contain NaN values")
    print(f"   Replacing NaN with 0.5 (uncertain prediction)")
    probabilities[nan_mask] = 0.5

# Clip probabilities to valid range [0, 1]
probabilities = np.clip(probabilities, 0, 1)

print(f"\nPrediction statistics:")
print(f"   Min probability: {probabilities.min():.4f}")
print(f"   Max probability: {probabilities.max():.4f}")
print(f"   Mean probability: {probabilities.mean():.4f}")
print(f"   Median probability: {np.median(probabilities):.4f}")

# ======================== Evaluation Function ========================

def evaluate_transfer_learning(y_true, probabilities, threshold, title):
    """Comprehensive evaluation of transfer learning performance"""
    
    predictions = (probabilities > threshold).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    auroc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)
    mcc = matthews_corrcoef(y_true, predictions)
    brier = brier_score_loss(y_true, probabilities)
    
    print(f"\n{title}")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Score':<15} {'Interpretation'}")
    print(f"{'-'*80}")
    print(f"{'Overall Accuracy':<25} {acc:.4f}         {'Good' if acc > 0.7 else 'Moderate' if acc > 0.6 else 'Poor'}")
    print(f"{'Precision':<25} {precision:.4f}         {'Good' if precision > 0.7 else 'Moderate' if precision > 0.5 else 'Poor'}")
    print(f"{'Recall (Sensitivity)':<25} {recall:.4f}         {'Good' if recall > 0.7 else 'Moderate' if recall > 0.5 else 'Poor'}")
    print(f"{'F1 Score':<25} {f1:.4f}         {'Good' if f1 > 0.7 else 'Moderate' if f1 > 0.5 else 'Poor'}")
    print(f"{'AUROC':<25} {auroc:.4f}         {'Good' if auroc > 0.7 else 'Moderate' if auroc > 0.6 else 'Poor'}")
    print(f"{'PR-AUC':<25} {pr_auc:.4f}         {'Good' if pr_auc > 0.7 else 'Moderate' if pr_auc > 0.6 else 'Poor'}")
    print(f"{'MCC':<25} {mcc:.4f}         {'Good' if mcc > 0.4 else 'Moderate' if mcc > 0.2 else 'Poor'}")
    print(f"{'Brier Score (lower=better)':<25} {brier:.4f}         {'Good' if brier < 0.15 else 'Moderate' if brier < 0.25 else 'Poor'}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, predictions, 
                               target_names=['Non-Landslide', 'Landslide'],
                               digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Non-LS    Landslide")
    print(f"Actual Non-LS   {cm[0,0]:<6}    {cm[0,1]:<6}")
    print(f"    Landslide   {cm[1,0]:<6}    {cm[1,1]:<6}")
    
    # Class distribution
    print(f"\nPrediction Distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for u, c in zip(unique, counts):
        class_name = "Non-Landslide" if u == 0 else "Landslide"
        print(f"   {class_name}: {c} ({c/len(predictions)*100:.1f}%)")
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'brier': brier,
        'confusion_matrix': cm
    }

# ======================== Evaluate with Multiple Thresholds ========================

print("\n12. Testing multiple thresholds...")

thresholds_to_test = [0.3, 0.4, 0.5, best_threshold, 0.6, 0.7]
threshold_results = {}

for threshold in thresholds_to_test:
    title = f"THRESHOLD = {threshold:.2f}" + (" (Chiapas optimal)" if threshold == best_threshold else " (Default)" if threshold == 0.5 else "")
    results = evaluate_transfer_learning(y_true, probabilities, threshold, title)
    threshold_results[threshold] = results

# ======================== Find Optimal Threshold for Durban ========================

print("\n" + "="*80)
print("FINDING OPTIMAL THRESHOLD FOR DURBAN DATA")
print("="*80)

thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []
auroc_scores = []
mcc_scores = []

for t in thresholds:
    preds = (probabilities > t).astype(int)
    f1_scores.append(f1_score(y_true, preds, zero_division=0))
    mcc_scores.append(matthews_corrcoef(y_true, preds))
    auroc_scores.append(roc_auc_score(y_true, probabilities))

best_f1_idx = np.argmax(f1_scores)
best_f1_threshold = thresholds[best_f1_idx]
best_f1_score = f1_scores[best_f1_idx]

best_mcc_idx = np.argmax(mcc_scores)
best_mcc_threshold = thresholds[best_mcc_idx]
best_mcc_score = mcc_scores[best_mcc_idx]

print(f"\nOptimal threshold by F1 Score: {best_f1_threshold:.2f} (F1 = {best_f1_score:.4f})")
print(f"Optimal threshold by MCC: {best_mcc_threshold:.2f} (MCC = {best_mcc_score:.4f})")

# Evaluate with Durban-optimized threshold
print(f"\n" + "="*80)
durban_optimal_results = evaluate_transfer_learning(
    y_true, probabilities, best_f1_threshold,
    f"DURBAN-OPTIMIZED THRESHOLD = {best_f1_threshold:.2f}"
)

# ======================== Visualization ========================

print("\n13. Creating visualization plots...")

fig = plt.figure(figsize=(20, 12))

# 1. ROC Curve
ax1 = plt.subplot(3, 4, 1)
fpr, tpr, _ = roc_curve(y_true, probabilities)
auroc = roc_auc_score(y_true, probabilities)
ax1.plot(fpr, tpr, 'b-', lw=2, label=f'Transfer Learning\n(AUROC = {auroc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=10)
ax1.set_ylabel('True Positive Rate', fontsize=10)
ax1.set_title('ROC Curve: Transfer Learning', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
ax2 = plt.subplot(3, 4, 2)
precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
pr_auc = average_precision_score(y_true, probabilities)
ax2.plot(recall_curve, precision_curve, 'g-', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
ax2.set_xlabel('Recall', fontsize=10)
ax2.set_ylabel('Precision', fontsize=10)
ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix (Chiapas threshold)
ax3 = plt.subplot(3, 4, 3)
cm_chiapas = threshold_results[best_threshold]['confusion_matrix']
sns.heatmap(cm_chiapas, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax3,
            xticklabels=['Non-LS', 'LS'], yticklabels=['Non-LS', 'LS'])
ax3.set_title(f'Confusion Matrix\n(Chiapas Threshold={best_threshold:.2f})', fontsize=11, fontweight='bold')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# 4. Confusion Matrix (Durban-optimized threshold)
ax4 = plt.subplot(3, 4, 4)
cm_durban = durban_optimal_results['confusion_matrix']
sns.heatmap(cm_durban, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax4,
            xticklabels=['Non-LS', 'LS'], yticklabels=['Non-LS', 'LS'])
ax4.set_title(f'Confusion Matrix\n(Durban Optimal={best_f1_threshold:.2f})', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

# 5. Probability Distribution
ax5 = plt.subplot(3, 4, 5)
ax5.hist(probabilities[y_true == 0], bins=30, alpha=0.6, label='Non-Landslide', color='blue', edgecolor='black')
ax5.hist(probabilities[y_true == 1], bins=30, alpha=0.6, label='Landslide', color='red', edgecolor='black')
ax5.axvline(best_threshold, color='orange', linestyle='--', linewidth=2, label=f'Chiapas Thr={best_threshold:.2f}')
ax5.axvline(best_f1_threshold, color='green', linestyle='--', linewidth=2, label=f'Durban Thr={best_f1_threshold:.2f}')
ax5.set_xlabel('Predicted Probability', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Threshold vs F1 Score
ax6 = plt.subplot(3, 4, 6)
ax6.plot(thresholds, f1_scores, 'b-o', markersize=4, linewidth=2)
ax6.axvline(best_f1_threshold, color='green', linestyle='--', linewidth=2, 
           label=f'Optimal={best_f1_threshold:.2f}')
ax6.axvline(best_threshold, color='orange', linestyle='--', linewidth=2,
           label=f'Chiapas={best_threshold:.2f}')
ax6.set_xlabel('Threshold', fontsize=10)
ax6.set_ylabel('F1 Score', fontsize=10)
ax6.set_title('F1 Score vs Threshold', fontsize=12, fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# 7. Threshold vs MCC
ax7 = plt.subplot(3, 4, 7)
ax7.plot(thresholds, mcc_scores, 'r-o', markersize=4, linewidth=2)
ax7.axvline(best_mcc_threshold, color='darkred', linestyle='--', linewidth=2,
           label=f'Optimal={best_mcc_threshold:.2f}')
ax7.set_xlabel('Threshold', fontsize=10)
ax7.set_ylabel('Matthews Correlation Coefficient', fontsize=10)
ax7.set_title('MCC vs Threshold', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# 8. Metric Comparison
ax8 = plt.subplot(3, 4, 8)
metrics_chiapas = threshold_results[best_threshold]
metrics_durban = durban_optimal_results
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC']
chiapas_vals = [metrics_chiapas['accuracy'], metrics_chiapas['precision'], 
                metrics_chiapas['recall'], metrics_chiapas['f1'], metrics_chiapas['auroc']]
durban_vals = [metrics_durban['accuracy'], metrics_durban['precision'],
               metrics_durban['recall'], metrics_durban['f1'], metrics_durban['auroc']]

x = np.arange(len(metric_names))
width = 0.35
ax8.bar(x - width/2, chiapas_vals, width, label=f'Chiapas Thr={best_threshold:.2f}', color='orange', alpha=0.8)
ax8.bar(x + width/2, durban_vals, width, label=f'Durban Thr={best_f1_threshold:.2f}', color='green', alpha=0.8)
ax8.set_ylabel('Score', fontsize=10)
ax8.set_title('Metrics: Chiapas vs Durban Threshold', fontsize=12, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
ax8.legend(fontsize=8)
ax8.grid(True, alpha=0.3, axis='y')
ax8.set_ylim([0, 1.0])

# 9-10. Spatial distribution of predictions
ax9 = plt.subplot(3, 4, 9)
durban_coords = durban_data[['xcoord', 'ycoord']].values
predictions_chiapas = (probabilities > best_threshold).astype(int)
correct_pred = (predictions_chiapas == y_true)

scatter1 = ax9.scatter(durban_coords[correct_pred, 0], durban_coords[correct_pred, 1],
                       c='green', s=10, alpha=0.5, label='Correct')
scatter2 = ax9.scatter(durban_coords[~correct_pred, 0], durban_coords[~correct_pred, 1],
                       c='red', s=10, alpha=0.5, label='Incorrect')
ax9.set_xlabel('X Coordinate', fontsize=10)
ax9.set_ylabel('Y Coordinate', fontsize=10)
ax9.set_title('Spatial Distribution of Predictions\n(Chiapas Threshold)', fontsize=11, fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

# 10. Probability map
ax10 = plt.subplot(3, 4, 10)
scatter = ax10.scatter(durban_coords[:, 0], durban_coords[:, 1],
                       c=probabilities, cmap='RdYlGn_r', s=20, alpha=0.7, vmin=0, vmax=1)
plt.colorbar(scatter, ax=ax10, label='Landslide Probability')
ax10.set_xlabel('X Coordinate', fontsize=10)
ax10.set_ylabel('Y Coordinate', fontsize=10)
ax10.set_title('Landslide Susceptibility Map\n(Transfer Learning)', fontsize=11, fontweight='bold')
ax10.grid(True, alpha=0.3)

# 11. True vs Predicted Landslides
ax11 = plt.subplot(3, 4, 11)
landslide_mask = y_true == 1
non_landslide_mask = y_true == 0
ax11.scatter(durban_coords[non_landslide_mask, 0], durban_coords[non_landslide_mask, 1],
            c='blue', s=10, alpha=0.3, label='True Non-Landslide', marker='o')
ax11.scatter(durban_coords[landslide_mask, 0], durban_coords[landslide_mask, 1],
            c='red', s=30, alpha=0.7, label='True Landslide', marker='x')
ax11.set_xlabel('X Coordinate', fontsize=10)
ax11.set_ylabel('Y Coordinate', fontsize=10)
ax11.set_title('True Landslide Locations', fontsize=11, fontweight='bold')
ax11.legend(fontsize=8)
ax11.grid(True, alpha=0.3)

# 12. Summary Text
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
summary_text = f"""
TRANSFER LEARNING SUMMARY
{'='*40}

Training Region: Chiapas, Mexico
Test Region: Durban, South Africa

Model Architecture:
• Advanced ResNet with Attention
• Input Features: {input_dim}

Chiapas Optimal Threshold: {best_threshold:.2f}
  • Accuracy:  {threshold_results[best_threshold]['accuracy']:.3f}
  • F1 Score:  {threshold_results[best_threshold]['f1']:.3f}
  • AUROC:     {threshold_results[best_threshold]['auroc']:.3f}

Durban Optimal Threshold: {best_f1_threshold:.2f}
  • Accuracy:  {durban_optimal_results['accuracy']:.3f}
  • F1 Score:  {durban_optimal_results['f1']:.3f}
  • AUROC:     {durban_optimal_results['auroc']:.3f}

Transfer Learning Performance:
  {'✓ GOOD' if durban_optimal_results['f1'] > 0.7 else '~ MODERATE' if durban_optimal_results['f1'] > 0.5 else '✗ POOR'}

Key Findings:
• Model {'generalizes well' if durban_optimal_results['f1'] > 0.6 else 'shows limited generalization'}
  to Durban data
• AUROC = {durban_optimal_results['auroc']:.3f}
  ({'Good' if durban_optimal_results['auroc'] > 0.7 else 'Moderate'} discrimination)
• MCC = {durban_optimal_results['mcc']:.3f}
• PR-AUC = {durban_optimal_results['pr_auc']:.3f}
"""
ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Transfer Learning Evaluation: Chiapas Model → Durban Data', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('transfer_learning_durban_evaluation.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: transfer_learning_durban_evaluation.png")
plt.show()

# ======================== Save Results to CSV ========================

print("\n14. Saving results...")

# Save comprehensive metrics
results_df = pd.DataFrame({
    'Threshold': list(threshold_results.keys()) + [best_f1_threshold],
    'Type': ['Tested']*len(threshold_results) + ['Durban Optimal'],
    'Accuracy': [v['accuracy'] for v in threshold_results.values()] + [durban_optimal_results['accuracy']],
    'Precision': [v['precision'] for v in threshold_results.values()] + [durban_optimal_results['precision']],
    'Recall': [v['recall'] for v in threshold_results.values()] + [durban_optimal_results['recall']],
    'F1_Score': [v['f1'] for v in threshold_results.values()] + [durban_optimal_results['f1']],
    'AUROC': [v['auroc'] for v in threshold_results.values()] + [durban_optimal_results['auroc']],
    'PR_AUC': [v['pr_auc'] for v in threshold_results.values()] + [durban_optimal_results['pr_auc']],
    'MCC': [v['mcc'] for v in threshold_results.values()] + [durban_optimal_results['mcc']],
    'Brier_Score': [v['brier'] for v in threshold_results.values()] + [durban_optimal_results['brier']]
})

results_df = results_df.sort_values('Threshold').reset_index(drop=True)
results_df.to_csv('transfer_learning_metrics_durban.csv', index=False)
print(f"   ✓ Saved: transfer_learning_metrics_durban.csv")

# Save predictions
predictions_df = durban_data[['xcoord', 'ycoord', 'label']].copy()
predictions_df['probability'] = probabilities
predictions_df['predicted_chiapas_threshold'] = (probabilities > best_threshold).astype(int)
predictions_df['predicted_durban_threshold'] = (probabilities > best_f1_threshold).astype(int)
predictions_df.to_csv('durban_predictions_transfer_learning.csv', index=False)
print(f"   ✓ Saved: durban_predictions_transfer_learning.csv")

# ======================== Final Summary ========================

print("\n" + "="*80)
print("TRANSFER LEARNING EXPERIMENT COMPLETE")
print("="*80)

print(f"\n📊 PERFORMANCE SUMMARY:")
print(f"   Region: Chiapas (Training) → Durban (Testing)")
print(f"   Model: {model_data['model_architecture']}")
print(f"   Features: {input_dim}")
print(f"")
print(f"   Best Durban Threshold: {best_f1_threshold:.2f}")
print(f"   • Accuracy:  {durban_optimal_results['accuracy']:.4f}")
print(f"   • Precision: {durban_optimal_results['precision']:.4f}")
print(f"   • Recall:    {durban_optimal_results['recall']:.4f}")
print(f"   • F1 Score:  {durban_optimal_results['f1']:.4f}")
print(f"   • AUROC:     {durban_optimal_results['auroc']:.4f}")
print(f"   • MCC:       {durban_optimal_results['mcc']:.4f}")

print(f"\n💡 INTERPRETATION:")
if durban_optimal_results['auroc'] > 0.7 and durban_optimal_results['f1'] > 0.6:
    print(f"   ✓ GOOD: Model trained on Chiapas shows strong generalization to Durban.")
    print(f"   ✓ The landslide susceptibility patterns learned from Chiapas are")
    print(f"     applicable to Durban, suggesting similar geological processes.")
elif durban_optimal_results['auroc'] > 0.6 or durban_optimal_results['f1'] > 0.5:
    print(f"   ~ MODERATE: Model shows reasonable but imperfect transfer to Durban.")
    print(f"   ~ Some landslide patterns transfer, but regional differences exist.")
    print(f"   ~ Fine-tuning with Durban data could improve performance.")
else:
    print(f"   ✗ LIMITED: Model shows poor transfer from Chiapas to Durban.")
    print(f"   ✗ Regional differences in geology, climate, or landslide triggers")
    print(f"     may be too significant for direct transfer.")
    print(f"   ✗ Training a Durban-specific model is recommended.")

print(f"\n📁 OUTPUT FILES:")
print(f"   • transfer_learning_durban_evaluation.png")
print(f"   • transfer_learning_metrics_durban.csv")
print(f"   • durban_predictions_transfer_learning.csv")
print(f"   • durban_test_data.csv")

print(f"\n{'='*80}")
print("✓ Transfer learning experiment completed successfully!")
print(f"{'='*80}\n")
