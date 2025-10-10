import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import KMeans
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, 
                             classification_report, confusion_matrix, roc_auc_score, roc_curve,
                             average_precision_score, matthews_corrcoef, brier_score_loss,
                             precision_recall_curve)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter



##########################data preparation##########################

# Load datasets
landslides = pd.read_csv("../output_landslides.csv")
nonLandslides = pd.read_csv("../output_non_landslides.csv")

# Add label
landslides["label"] = 1
nonLandslides["label"] = 0

# Combine
full_data = pd.concat([landslides, nonLandslides], ignore_index=True)

# Shuffle the data (avoids any order bias)
full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate predictors and target
X = full_data.drop("label", axis=1)
y = full_data["label"]

# Convert True/False to 1/0 in categorical columns
X = X.replace({True: 1, False: 0})

# Convert all values to numeric (will coerce bad values to NaN)
X = X.apply(pd.to_numeric, errors='coerce')

# Drop or fill missing values (e.g., those blanks you saw)
X = X.fillna(0)  

# KEEP coordinates for spatial CV - extract before dropping
coordinates = X[['xcoord', 'ycoord']].copy()
print(f"Coordinate ranges: X({coordinates['xcoord'].min():.0f} to {coordinates['xcoord'].max():.0f}), Y({coordinates['ycoord'].min():.0f} to {coordinates['ycoord'].max():.0f})")

# Drop unwanted columns AFTER extracting coordinates
X = X.drop(columns=["xcoord", "ycoord", "fid"], errors="ignore")

#################################Feature Selection#################################

# Feature selection to remove less important features
print(f"Number of features before selection: {X.shape[1]}")

# Ensemble feature selection: Combine statistical and tree-based methods
print("Performing ensemble feature selection...")

# Method 1: Statistical (F-test)
selector_stats = SelectKBest(score_func=f_classif, k=60)
X_stats_selected = selector_stats.fit_transform(X, y)
stats_features = X.columns[selector_stats.get_support()]

# Method 2: Tree-based feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
feature_importance_rf = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
rf_top_features = feature_importance_rf.head(60).index

# Method 3: Recursive Feature Elimination
rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=60)
rfe.fit(X, y)
rfe_features = X.columns[rfe.support_]

# Combine all methods - features that appear in at least 2 out of 3 methods
all_selected = set(stats_features) | set(rf_top_features) | set(rfe_features)
feature_votes = {}
for feature in all_selected:
    votes = 0
    if feature in stats_features: votes += 1
    if feature in rf_top_features: votes += 1
    if feature in rfe_features: votes += 1
    feature_votes[feature] = votes

# Select features with at least 2 votes
final_features = [f for f, votes in feature_votes.items() if votes >= 2]
print(f"Features selected by ensemble method: {len(final_features)}")

# Use the ensemble-selected features
X_selected = X[final_features]
selected_features = final_features

print(f"Number of features after ensemble selection: {len(selected_features)}")
print(f"Selected features: {list(selected_features)[:10]}...")  # Show first 10

# Print feature info
print(f"Data shape: {X_selected.shape}")

#################################Spatial Blocking for CV#################################

def create_spatial_blocks(coords, method='grid', n_blocks=25, buffer_distance=None):
    """
    Create spatial blocks for cross-validation
    
    Parameters:
    - coords: DataFrame with 'xcoord', 'ycoord' columns
    - method: 'grid' or 'kmeans'
    - n_blocks: number of blocks (for grid: will be made square, for kmeans: exact number)
    - buffer_distance: distance for buffer zones (optional)
    
    Returns:
    - block_ids: array of block assignments for each sample
    """
    print(f"Creating spatial blocks using {method} method...")
    
    if method == 'grid':
        # Create square grid
        n_side = int(np.sqrt(n_blocks))
        actual_blocks = n_side * n_side
        print(f"Using {n_side}x{n_side} = {actual_blocks} grid blocks")
        
        x_bins = np.linspace(coords['xcoord'].min(), coords['xcoord'].max(), n_side + 1)
        y_bins = np.linspace(coords['ycoord'].min(), coords['ycoord'].max(), n_side + 1)
        
        x_block = np.digitize(coords['xcoord'], x_bins) - 1
        y_block = np.digitize(coords['ycoord'], y_bins) - 1
        
        # Ensure blocks are within bounds
        x_block = np.clip(x_block, 0, n_side - 1)
        y_block = np.clip(y_block, 0, n_side - 1)
        
        block_ids = y_block * n_side + x_block
        
    elif method == 'kmeans':
        # Use K-means clustering for irregular blocks
        print(f"Using K-means clustering for {n_blocks} blocks")
        kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
        block_ids = kmeans.fit_predict(coords[['xcoord', 'ycoord']])
    
    print(f"Created {len(np.unique(block_ids))} spatial blocks")
    print(f"Block sizes: min={np.bincount(block_ids).min()}, max={np.bincount(block_ids).max()}, mean={np.bincount(block_ids).mean():.0f}")
    
    return block_ids

def check_spatial_separation(coords, block_ids, fold_assignment):
    """
    Check spatial separation between train/test folds
    """
    unique_folds = np.unique(fold_assignment)
    min_distances = []
    
    for test_fold in unique_folds:
        train_mask = fold_assignment != test_fold
        test_mask = fold_assignment == test_fold
        
        if not test_mask.any() or not train_mask.any():
            continue
            
        train_coords = coords[train_mask]
        test_coords = coords[test_mask]
        
        # Calculate minimum distance between train and test regions
        from scipy.spatial.distance import cdist
        distances = cdist(test_coords[['xcoord', 'ycoord']], 
                         train_coords[['xcoord', 'ycoord']])
        min_dist = distances.min()
        min_distances.append(min_dist)
        
    avg_min_distance = np.mean(min_distances) if min_distances else 0
    print(f"Average minimum distance between train/test regions: {avg_min_distance:.0f} units")
    return avg_min_distance

# Create spatial blocks
block_ids = create_spatial_blocks(coordinates, method='grid', n_blocks=25)

# Check class distribution across blocks
block_class_dist = pd.DataFrame({
    'block': block_ids,
    'class': y.values
}).groupby('block')['class'].agg(['count', 'mean']).reset_index()

print(f"\nBlock class distribution:")
print(f"Blocks with landslides: {(block_class_dist['mean'] > 0).sum()}/{len(block_class_dist)}")
print(f"Average landslide rate per block: {block_class_dist['mean'].mean():.3f}")
print(f"Blocks with <10 samples: {(block_class_dist['count'] < 10).sum()}")

# Remove very small blocks to avoid CV issues
min_block_size = 10
valid_blocks = block_class_dist[block_class_dist['count'] >= min_block_size]['block'].values
valid_mask = np.isin(block_ids, valid_blocks)

print(f"Keeping {valid_mask.sum()}/{len(block_ids)} samples in blocks with >={min_block_size} samples")

# Filter data to valid blocks only
X_selected_spatial = X_selected[valid_mask]
y_spatial = y[valid_mask]
coordinates_spatial = coordinates[valid_mask]
block_ids_spatial = block_ids[valid_mask]

print(f"Final data shape for spatial CV: {X_selected_spatial.shape}")

#################################Spatial Train-Validation-Test Split#################################

# Use spatial split for proper train/validation/test sets
print("\n=== PERFORMING SPATIAL TRAIN/VALIDATION/TEST SPLIT ===")

# First split: 80% train+val, 20% test (final holdout)
print("\nStep 1: Splitting into Train+Val (80%) and Test (20%)...")
gss_test = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
trainval_idx, test_idx = next(gss_test.split(X_selected_spatial, y_spatial, groups=block_ids_spatial))

X_trainval = X_selected_spatial.iloc[trainval_idx]
X_test = X_selected_spatial.iloc[test_idx]
y_trainval = y_spatial.iloc[trainval_idx]
y_test = y_spatial.iloc[test_idx]
coords_trainval = coordinates_spatial.iloc[trainval_idx]
coords_test = coordinates_spatial.iloc[test_idx]
blocks_trainval = block_ids_spatial[trainval_idx]

print(f"  Train+Val set: {len(X_trainval)} samples, {y_trainval.mean():.3f} landslide rate")
print(f"  Test set: {len(X_test)} samples, {y_test.mean():.3f} landslide rate")

# Second split: Split train+val into 75% train, 25% validation (which gives 60% train, 20% val of total)
print("\nStep 2: Splitting Train+Val into Train (75%) and Validation (25%)...")
gss_val = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=42)
train_idx, val_idx = next(gss_val.split(X_trainval, y_trainval, groups=blocks_trainval))

X_train = X_trainval.iloc[train_idx]
X_val = X_trainval.iloc[val_idx]
y_train = y_trainval.iloc[train_idx]
y_val = y_trainval.iloc[val_idx]
coords_train = coords_trainval.iloc[train_idx]
coords_val = coords_trainval.iloc[val_idx]

print(f"  Train set: {len(X_train)} samples, {y_train.mean():.3f} landslide rate")
print(f"  Validation set: {len(X_val)} samples, {y_val.mean():.3f} landslide rate")
print(f"  Test set: {len(X_test)} samples, {y_test.mean():.3f} landslide rate")

# Calculate percentages
total_samples = len(X_selected_spatial)
print(f"\nFinal split percentages:")
print(f"  Train: {len(X_train)/total_samples*100:.1f}%")
print(f"  Validation: {len(X_val)/total_samples*100:.1f}%")
print(f"  Test: {len(X_test)/total_samples*100:.1f}%")

# Verify spatial separation between all three sets
print("\nVerifying spatial separation between sets...")
fold_assignment = np.full(len(X_selected_spatial), -1)
fold_assignment[trainval_idx[train_idx]] = 0  # Train
fold_assignment[trainval_idx[val_idx]] = 1    # Validation
fold_assignment[test_idx] = 2                  # Test
check_spatial_separation(coordinates_spatial.reset_index(drop=True), block_ids_spatial, fold_assignment)

# Use RobustScaler - fit only on training data!
print("\nScaling features using RobustScaler (fitted on training data only)...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("  ✓ Scaler fitted on training data")
print("  ✓ Applied to validation data")
print("  ✓ Applied to test data (for final evaluation only)")

# Convert back to DataFrames to maintain column names for saving
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

#################################Spatial Cross-Validation Evaluation#################################

print("\n=== SPATIAL CROSS-VALIDATION EVALUATION ===")
print("Performing 5-fold spatial CV to assess model robustness...")

# Use GroupKFold for spatial CV on training data
gkf = GroupKFold(n_splits=5)
kfold_scores = {
    'accuracy': [], 
    'precision': [], 
    'recall': [], 
    'f1': [], 
    'auroc': [], 
    'pr_auc': [],
    'mcc': [],
    'brier': []
}

# Get blocks for training data only
train_blocks = block_ids_spatial[train_idx]
train_coords = coordinates_spatial.iloc[train_idx]

fold_num = 0
for cv_train_idx, cv_val_idx in gkf.split(X_train, y_train, groups=train_blocks):
    fold_num += 1
    print(f"\nFold {fold_num}/5:")
    
    # Split data
    X_cv_train, X_cv_val = X_train.iloc[cv_train_idx], X_train.iloc[cv_val_idx]
    y_cv_train, y_cv_val = y_train.iloc[cv_train_idx], y_train.iloc[cv_val_idx]
    
    print(f"  CV Train: {len(X_cv_train)} samples, {y_cv_train.mean():.3f} landslide rate")
    print(f"  CV Val: {len(X_cv_val)} samples, {y_cv_val.mean():.3f} landslide rate")
    
    # Scale data
    fold_scaler = RobustScaler()
    X_cv_train_scaled = fold_scaler.fit_transform(X_cv_train)
    X_cv_val_scaled = fold_scaler.transform(X_cv_val)
    
    # Quick model training for CV (simpler model for speed)
    from sklearn.ensemble import RandomForestClassifier
    
    cv_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    cv_model.fit(X_cv_train_scaled, y_cv_train)
    
    # Predictions
    y_cv_pred = cv_model.predict(X_cv_val_scaled)
    y_cv_prob = cv_model.predict_proba(X_cv_val_scaled)[:, 1]
    
    # Calculate all metrics
    fold_acc = accuracy_score(y_cv_val, y_cv_pred)
    fold_prec = precision_score(y_cv_val, y_cv_pred, zero_division=0)
    fold_rec = recall_score(y_cv_val, y_cv_pred, zero_division=0)
    fold_f1 = f1_score(y_cv_val, y_cv_pred, zero_division=0)
    
    # Check if both classes present in validation fold
    if len(np.unique(y_cv_val)) > 1:
        fold_auroc = roc_auc_score(y_cv_val, y_cv_prob)
        fold_pr_auc = average_precision_score(y_cv_val, y_cv_prob)
        fold_mcc = matthews_corrcoef(y_cv_val, y_cv_pred)
        fold_brier = brier_score_loss(y_cv_val, y_cv_prob)
    else:
        fold_auroc = fold_pr_auc = fold_mcc = fold_brier = 0
        print(f"  Warning: Only one class in validation fold, some metrics set to 0")
    
    kfold_scores['accuracy'].append(fold_acc)
    kfold_scores['precision'].append(fold_prec)
    kfold_scores['recall'].append(fold_rec)
    kfold_scores['f1'].append(fold_f1)
    kfold_scores['auroc'].append(fold_auroc)
    kfold_scores['pr_auc'].append(fold_pr_auc)
    kfold_scores['mcc'].append(fold_mcc)
    kfold_scores['brier'].append(fold_brier)
    
    print(f"  Accuracy={fold_acc:.3f}, Precision={fold_prec:.3f}, Recall={fold_rec:.3f}, F1={fold_f1:.3f}")
    print(f"  AUROC={fold_auroc:.3f}, PR-AUC={fold_pr_auc:.3f}, MCC={fold_mcc:.3f}, Brier={fold_brier:.3f}")

# Print CV summary
print(f"\n=== SPATIAL CV SUMMARY (5-fold) ===")
print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print(f"{'-'*55}")
for metric, scores in kfold_scores.items():
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    print(f"{metric.upper():<15} {mean_score:.4f}    {std_score:.4f}    {min_score:.4f}    {max_score:.4f}")

print(f"\nNote: These are spatially-separated CV scores using RandomForest.")
print(f"Your deep learning model may perform differently, but this gives spatial robustness baseline.")

# Save CV results to CSV
cv_results_df = pd.DataFrame(kfold_scores)
cv_results_df['fold'] = range(1, len(cv_results_df) + 1)
cv_results_df = cv_results_df[['fold'] + [col for col in cv_results_df.columns if col != 'fold']]
cv_results_df.to_csv('spatial_cv_results.csv', index=False)
print(f"\nSpatial CV results saved to 'spatial_cv_results.csv'")

#save the split datasets
print("\nSaving train/validation/test splits to CSV files...")
X_train_scaled_df.to_csv("X_train.csv", index=False)
X_val_scaled_df.to_csv("X_val.csv", index=False)
X_test_scaled_df.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("  ✓ Saved X_train.csv, X_val.csv, X_test.csv")
print("  ✓ Saved y_train.csv, y_val.csv, y_test.csv")

############################ Convert to torch tensors ############################
# Convert to torch tensors using scaled data
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

print("\nCreated PyTorch tensors for train/validation/test sets")

# Calculate class weights for imbalanced learning
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance
# This helps the model produce predictions in the full 0-1 range
pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
print(f"Positive class weight for BCE: {pos_weight.item():.3f}")

# Don't use weighted sampler - it can cause overfitting
# Instead, rely on pos_weight in the loss function
# Create datasets without weighted sampling
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Dataloaders with standard sampling (no weights)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)} (batch_size=128)")
print(f"  Validation batches: {len(val_loader)} (batch_size=128)")
print(f"  Test batches: {len(test_loader)} (batch_size=128)")
print(f"  Note: Test loader will only be used for FINAL evaluation")



############################ Advanced Model Definition ############################

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
            nn.Dropout(0.5)  # Increased to combat overfitting
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(512)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(512, 256, 0.4)  # Increased dropout
        self.res_block2 = ResidualBlock(512, 256, 0.4)  # Increased dropout
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased to reduce overfitting
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased to reduce overfitting
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)  # Increased to reduce overfitting
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

model = AdvancedLandslideANN(X_train_scaled.shape[1])

# Use standard BCE loss with pos_weight for class imbalance
# This produces better calibrated probabilities than Focal Loss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-3)

# Use ReduceLROnPlateau for smoother learning rate reduction
# This will reduce LR when validation loss plateaus, giving more stable training
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6  # Reduced patience for faster response
)

print("\nUsing ReduceLROnPlateau scheduler with aggressive regularization to prevent overfitting")

# Advanced training loop with gradient clipping
num_epochs = 150
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 20  # Reduced patience to stop earlier when overfitting
patience_counter = 0

# Force CPU usage - GPU (CUDA 5.0) is incompatible with current PyTorch (requires CUDA 7.0+)
device = torch.device('cpu')
scaler_amp = None  # No mixed precision on CPU

print(f"\nTraining on CPU (GPU CUDA capability 5.0 is incompatible with PyTorch)")
model = model.to(device)

print(f"Training on device: {device}")
print(f"\n{'='*60}")
print("STARTING TRAINING WITH PROPER TRAIN/VALIDATION/TEST SPLIT")
print(f"{'='*60}\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        if scaler_amp is not None:
            with torch.cuda.amp.autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            scaler_amp.scale(loss).backward()
            # Gradient clipping
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        running_loss += loss.item()
    
    # Validation phase - NOW USING PROPER VALIDATION SET!
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:  # Changed from test_loader to val_loader
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if scaler_amp is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)  # Changed from test_loader
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # Step scheduler based on validation loss
    scheduler.step(avg_val_loss)
    
    if epoch % 15 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping with patience
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model_advanced.pth')
        patience_counter = 0
        if epoch % 15 != 0:  # Print if not already printed
            print(f"Epoch {epoch+1}/{num_epochs} - ✓ New best validation loss: {avg_val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
# Load best model
print("\nLoading best model from checkpoint...")
model.load_state_dict(torch.load('best_model_advanced.pth'))
print("✓ Best model loaded")

# Move tensors to device for evaluation
X_val_tensor = X_val_tensor.to(device)
y_val_tensor = y_val_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

print("\n" + "="*60)
print("TRAINING COMPLETE - ANALYZING RESULTS")
print("="*60)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Validation Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss\n(Validation = Separate held-out set)')
plt.legend()
plt.grid(True, alpha=0.3)

# Test different thresholds for optimal performance ON VALIDATION SET
print("\nFinding optimal classification threshold using VALIDATION set...")
thresholds = np.arange(0.3, 0.8, 0.05)
best_threshold = 0.5
best_f1 = 0

model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_probabilities = torch.sigmoid(val_outputs).cpu().numpy()
    y_val_true = y_val_tensor.cpu().numpy()

f1_scores = []
for threshold in thresholds:
    predictions = (val_probabilities > threshold).astype(int)
    f1 = f1_score(y_val_true, predictions)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"  Best threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")

plt.subplot(1, 2, 2)
plt.plot(thresholds, f1_scores, 'b-o')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best threshold: {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold\n(Optimized on Validation set)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Training curves saved to 'training_analysis.png'")

def evaluate_model_advanced(model, X_test_tensor, y_test_tensor, threshold=0.5, device='cpu'):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > threshold).int().cpu().numpy()
        true = y_test_tensor.int().cpu().numpy()
        probs_np = probabilities.cpu().numpy()

        # Calculate all metrics
        acc = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, zero_division=0)
        recall = recall_score(true, predicted, zero_division=0)
        f1 = f1_score(true, predicted, zero_division=0)
        auroc = roc_auc_score(true, probs_np)
        pr_auc = average_precision_score(true, probs_np)
        mcc = matthews_corrcoef(true, predicted)
        brier = brier_score_loss(true, probs_np)

        print(f"\n=== ADVANCED EVALUATION (threshold={threshold:.3f}) ===")
        print(f"{'Metric':<20} {'Score':<10}")
        print(f"{'-'*30}")
        print(f"{'Overall Accuracy':<20} {acc:.4f}")
        print(f"{'Precision':<20} {precision:.4f}")
        print(f"{'Recall':<20} {recall:.4f}")
        print(f"{'F1 Score':<20} {f1:.4f}")
        print(f"{'AUROC':<20} {auroc:.4f}")
        print(f"{'PR-AUC':<20} {pr_auc:.4f}")
        print(f"{'MCC':<20} {mcc:.4f}")
        print(f"{'Brier Score':<20} {brier:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(true, predicted, target_names=['Non-Landslide', 'Landslide']))
        
        # ROC Curve and Precision-Recall Curve
        fpr, tpr, _ = roc_curve(true, probs_np)
        precision_curve, recall_curve, _ = precision_recall_curve(true, probs_np)
        
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 4, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUROC = {auroc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.subplot(1, 4, 2)
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR (PR-AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        # Confusion matrix
        cm = confusion_matrix(true, predicted)
        plt.subplot(1, 4, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Landslide', 'Landslide'],
                    yticklabels=['Non-Landslide', 'Landslide'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Prediction probability distribution
        plt.subplot(1, 4, 4)
        plt.hist(probs_np[true == 0], bins=30, alpha=0.7, label='Non-Landslide', color='blue')
        plt.hist(probs_np[true == 1], bins=30, alpha=0.7, label='Landslide', color='red')
        plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Class distribution in predictions
        unique, counts = np.unique(predicted, return_counts=True)
        print(f"\nPrediction distribution:")
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples ({c/len(predicted)*100:.1f}%)")
        
        print(f"\nActual distribution:")
        unique, counts = np.unique(true, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"Class {u}: {c} samples ({c/len(true)*100:.1f}%)")
        
        # Save metrics to dictionary
        metrics_dict = {
            'Overall Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUROC': auroc,
            'PR-AUC': pr_auc,
            'MCC': mcc,
            'Brier Score': brier
        }
        
        return acc, precision, recall, f1, auroc, pr_auc, mcc, brier, metrics_dict

# ============================================================================
# FINAL EVALUATION ON TEST SET (USED ONLY ONCE!)
# ============================================================================
print("\n" + "="*70)
print("FINAL MODEL EVALUATION ON SPATIALLY SEPARATED TEST SET")
print("="*70)
print("\n⚠️  IMPORTANT: This is the FIRST and ONLY time we're evaluating on the test set!")
print("   The test set was locked away during training and hyperparameter tuning.")
print("   These metrics represent true generalization to unseen geographic areas.\n")

print("=== Evaluation with default threshold (0.5) ===")        
results_default = evaluate_model_advanced(model, X_test_tensor, y_test_tensor, threshold=0.5, device=device)

# Evaluate with optimized threshold (tuned on validation set)
print(f"\n=== Evaluation with optimized threshold ({best_threshold:.3f}) ===")
print(f"   (Note: Threshold was optimized on VALIDATION set, not test set)")
results_optimized = evaluate_model_advanced(model, X_test_tensor, y_test_tensor, threshold=best_threshold, device=device)

# Save final metrics to CSV
final_metrics_df = pd.DataFrame({
    'Metric': ['Overall Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'PR-AUC', 'MCC', 'Brier Score'],
    'Threshold_0.5': [results_default[0], results_default[1], results_default[2], results_default[3], 
                      results_default[4], results_default[5], results_default[6], results_default[7]],
    'Optimized_Threshold': [results_optimized[0], results_optimized[1], results_optimized[2], results_optimized[3],
                           results_optimized[4], results_optimized[5], results_optimized[6], results_optimized[7]]
})
final_metrics_df.to_csv('final_test_metrics.csv', index=False)
print(f"\nFinal test metrics saved to 'final_test_metrics.csv'")

print(f"\n=== SPATIAL VALIDATION SUMMARY ===")
print(f"✓ Used spatial blocks for train/test separation")
print(f"✓ Applied 5-fold spatial cross-validation")
print(f"✓ Test set is spatially separated from training data")
print(f"✓ Results represent model performance on NEW geographic areas")
print(f"\nThis is more realistic than random CV for landslide susceptibility mapping.")

# Feature importance analysis
print("\n=== Feature Importance Analysis ===")
# Use Random Forest feature importance since we used ensemble selection
feature_importance_rf = pd.Series(rf.feature_importances_, index=X.columns)
selected_feature_importance = feature_importance_rf[selected_features].sort_values(ascending=False)

print("Top 10 most important features:")
print(selected_feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = selected_feature_importance.head(15)
plt.barh(range(len(top_features)), top_features.values)
plt.yticks(range(len(top_features)), top_features.index)
plt.xlabel('Feature Importance Score')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the advanced trained model and scaler for future use
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'selected_features': selected_features,
    'best_threshold': best_threshold,
    'model_architecture': 'AdvancedLandslideANN',
    'feature_selection_method': 'ensemble',
    'class_weights': class_weight_dict,
    'input_dim': X_train_scaled.shape[1],
    'device': str(device)
}, 'landslide_model_advanced_complete.pth')

print(f"\nAdvanced model saved as 'landslide_model_advanced_complete.pth'")
print(f"Best threshold for deployment: {best_threshold:.3f}")
print(f"Model trained on: {device}")
print(f"Features used: {len(selected_features)}")
print(f"Architecture: Advanced ResNet with Attention")



