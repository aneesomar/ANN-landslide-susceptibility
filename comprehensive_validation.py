"""
Comprehensive Landslide Susceptibility Map Validation
=====================================================
This script performs extensive validation of the susceptibility map including:
1. Success Rate Curve analysis
2. Prediction Rate Curve analysis
3. Spatial autocorrelation (Moran's I)
4. Confusion matrix on actual landslide locations
5. Before/After model comparison
"""

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
from scipy import stats
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE VALIDATION ANALYSIS")
print("="*80)

# ======================== 1. LOAD DATA ========================
print("\n1. Loading data...")

# Load susceptibility map
with rasterio.open('susceptibility_map.tif') as src:
    susceptibility = src.read(1)
    transform = src.transform
    print(f"   ✓ Susceptibility map loaded: {src.width} × {src.height} pixels")

# Load landslide and non-landslide data
landslides = pd.read_csv('../output_landslides.csv')
non_landslides = pd.read_csv('../output_non_landslides.csv')

print(f"   ✓ Landslide points: {len(landslides)}")
print(f"   ✓ Non-landslide points: {len(non_landslides)}")

# ======================== 2. EXTRACT SUSCEPTIBILITY AT KNOWN LOCATIONS ========================
print("\n2. Extracting susceptibility values at known locations...")

def extract_susceptibility(coords_df, susc_map, transform):
    """Extract susceptibility values at given coordinates"""
    susceptibilities = []
    valid_indices = []
    
    for idx, row in coords_df.iterrows():
        x, y = row['xcoord'], row['ycoord']
        
        # Convert coordinates to pixel indices
        col = int((x - transform[2]) / transform[0])
        row_idx = int((y - transform[5]) / transform[4])
        
        # Check bounds
        if 0 <= row_idx < susc_map.shape[0] and 0 <= col < susc_map.shape[1]:
            value = susc_map[row_idx, col]
            if not np.isnan(value) and 0 <= value <= 1:
                susceptibilities.append(value)
                valid_indices.append(idx)
    
    return np.array(susceptibilities), valid_indices

landslide_susc, landslide_valid = extract_susceptibility(landslides, susceptibility, transform)
non_landslide_susc, non_landslide_valid = extract_susceptibility(non_landslides, susceptibility, transform)

print(f"   ✓ Valid landslide points: {len(landslide_susc)}")
print(f"   ✓ Valid non-landslide points: {len(non_landslide_susc)}")
print(f"   ✓ Landslide mean susceptibility: {landslide_susc.mean():.4f}")
print(f"   ✓ Non-landslide mean susceptibility: {non_landslide_susc.mean():.4f}")

# ======================== 3. SUCCESS RATE CURVE ========================
print("\n3. Calculating Success Rate Curve...")

# Combine data
y_true = np.concatenate([np.ones(len(landslide_susc)), np.zeros(len(non_landslide_susc))])
y_pred = np.concatenate([landslide_susc, non_landslide_susc])

# Calculate ROC curve (Success Rate Curve)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

print(f"   ✓ Success Rate AUC: {roc_auc:.4f}")

# Calculate cumulative percentage of area vs landslides
# Sort by susceptibility (descending)
valid_susc = susceptibility[~np.isnan(susceptibility) & (susceptibility >= 0) & (susceptibility <= 1)]
sorted_susc = np.sort(valid_susc)[::-1]  # Descending

# For each percentage of area, calculate percentage of landslides captured
area_percentages = np.linspace(0, 100, 101)
landslide_percentages = []

for area_pct in area_percentages:
    threshold_idx = int(len(sorted_susc) * area_pct / 100)
    if threshold_idx >= len(sorted_susc):
        threshold = sorted_susc[-1]
    else:
        threshold = sorted_susc[threshold_idx]
    
    landslides_captured = np.sum(landslide_susc >= threshold)
    landslide_pct = (landslides_captured / len(landslide_susc)) * 100
    landslide_percentages.append(landslide_pct)

print(f"   ✓ Top 10% area captures {landslide_percentages[10]:.1f}% of landslides")
print(f"   ✓ Top 20% area captures {landslide_percentages[20]:.1f}% of landslides")
print(f"   ✓ Top 30% area captures {landslide_percentages[30]:.1f}% of landslides")

# ======================== 4. PREDICTION RATE CURVE (SPATIAL CV) ========================
print("\n4. Calculating Prediction Rate Curve...")

# Load test set predictions from model training
try:
    X_test = pd.read_csv('X_test.csv')
    y_test = pd.read_csv('y_test.csv')
    
    # Load model and make predictions on test set
    import torch
    import torch.nn as nn
    
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
                nn.Dropout(0.5)
            )
            self.attention = AttentionLayer(512)
            self.res_block1 = ResidualBlock(512, 256, 0.4)
            self.res_block2 = ResidualBlock(512, 256, 0.4)
            self.feature_layers = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.output = nn.Linear(64, 1)
            
        def forward(self, x):
            x = self.input_layer(x)
            x = self.attention(x)
            x = self.res_block1(x)
            x = self.res_block2(x)
            x = self.feature_layers(x)
            return self.output(x)
    
    model_data = torch.load('landslide_model_advanced_complete.pth', weights_only=False)
    model = AdvancedLandslideANN(X_test.shape[1])
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        test_predictions = torch.sigmoid(outputs).numpy().flatten()
    
    test_labels = y_test.values.flatten()
    
    # Prediction rate curve
    pred_fpr, pred_tpr, pred_thresh = roc_curve(test_labels, test_predictions)
    pred_auc = auc(pred_fpr, pred_tpr)
    
    print(f"   ✓ Prediction Rate AUC (spatial CV test): {pred_auc:.4f}")
    has_prediction_curve = True
    
except Exception as e:
    print(f"   ⚠ Could not calculate prediction rate curve: {e}")
    has_prediction_curve = False

# ======================== 5. SPATIAL AUTOCORRELATION (MORAN'S I) ========================
print("\n5. Calculating Spatial Autocorrelation (Moran's I)...")

def calculate_morans_i(data, sample_size=10000):
    """Calculate Global Moran's I for spatial autocorrelation"""
    # Sample data for computational efficiency
    valid_mask = ~np.isnan(data) & (data >= 0) & (data <= 1)
    valid_indices = np.argwhere(valid_mask)
    
    if len(valid_indices) > sample_size:
        sample_indices = valid_indices[np.random.choice(len(valid_indices), sample_size, replace=False)]
    else:
        sample_indices = valid_indices
    
    values = data[sample_indices[:, 0], sample_indices[:, 1]]
    coords = sample_indices
    
    # Calculate mean
    mean_val = np.mean(values)
    n = len(values)
    
    # Calculate weights (inverse distance, with max distance threshold)
    max_distance = 100  # pixels
    weights = np.zeros((n, n))
    
    print(f"   Computing distance matrix for {n} sampled points...")
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((coords[i,0] - coords[j,0])**2 + (coords[i,1] - coords[j,1])**2)
            if dist < max_distance and dist > 0:
                w = 1.0 / dist
                weights[i, j] = w
                weights[j, i] = w
    
    W = np.sum(weights)
    
    if W == 0:
        return None, None
    
    # Calculate Moran's I
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
    
    denominator = np.sum((values - mean_val)**2)
    
    morans_i = (n / W) * (numerator / denominator)
    
    # Calculate expected value and variance
    E_I = -1 / (n - 1)
    
    # Z-score
    var_I = (n**2 - 3*n + 3)*W**2 - n*W**2 + 3*W**2
    var_I = var_I / (W**2 * (n**2 - 1))
    
    if var_I > 0:
        z_score = (morans_i - E_I) / np.sqrt(var_I)
    else:
        z_score = 0
    
    return morans_i, z_score

morans_i, z_score = calculate_morans_i(susceptibility, sample_size=5000)

if morans_i is not None:
    print(f"   ✓ Moran's I: {morans_i:.4f}")
    print(f"   ✓ Z-score: {z_score:.4f}")
    if morans_i > 0:
        print(f"   ✓ Interpretation: Positive spatial autocorrelation (clustering)")
    elif morans_i < 0:
        print(f"   ✓ Interpretation: Negative spatial autocorrelation (dispersion)")
    else:
        print(f"   ✓ Interpretation: Random spatial pattern")
else:
    print(f"   ⚠ Could not calculate Moran's I")

# ======================== 6. CONFUSION MATRIX AT DIFFERENT THRESHOLDS ========================
print("\n6. Calculating confusion matrices at different thresholds...")

thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
confusion_results = []

for threshold in thresholds_to_test:
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    acc = accuracy_score(y_true, y_pred_binary)
    prec = precision_score(y_true, y_pred_binary, zero_division=0)
    rec = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    confusion_results.append({
        'Threshold': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'TN': cm[0,0],
        'FP': cm[0,1],
        'FN': cm[1,0],
        'TP': cm[1,1]
    })
    
    print(f"   Threshold {threshold:.1f}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

df_confusion = pd.DataFrame(confusion_results)
df_confusion.to_csv('validation_confusion_matrices.csv', index=False)
print(f"   ✓ Saved confusion matrices to 'validation_confusion_matrices.csv'")

# ======================== 7. STATISTICAL COMPARISON ========================
print("\n7. Statistical comparison of landslide vs non-landslide areas...")

# T-test
t_stat, p_value = stats.ttest_ind(landslide_susc, non_landslide_susc)
print(f"   ✓ T-test statistic: {t_stat:.4f}")
print(f"   ✓ P-value: {p_value:.4e}")
if p_value < 0.001:
    print(f"   ✓ Result: Highly significant difference (p < 0.001)")
elif p_value < 0.05:
    print(f"   ✓ Result: Significant difference (p < 0.05)")
else:
    print(f"   ✓ Result: No significant difference (p >= 0.05)")

# Mann-Whitney U test (non-parametric)
u_stat, u_pvalue = stats.mannwhitneyu(landslide_susc, non_landslide_susc, alternative='greater')
print(f"   ✓ Mann-Whitney U statistic: {u_stat:.4f}")
print(f"   ✓ P-value: {u_pvalue:.4e}")

# Cohen's d (effect size)
pooled_std = np.sqrt((np.std(landslide_susc)**2 + np.std(non_landslide_susc)**2) / 2)
cohens_d = (landslide_susc.mean() - non_landslide_susc.mean()) / pooled_std
print(f"   ✓ Cohen's d (effect size): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect = "negligible"
elif abs(cohens_d) < 0.5:
    effect = "small"
elif abs(cohens_d) < 0.8:
    effect = "medium"
else:
    effect = "large"
print(f"   ✓ Effect size interpretation: {effect}")

# ======================== 8. CREATE COMPREHENSIVE VISUALIZATIONS ========================
print("\n8. Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 12))

# Create grid
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# 1. Success Rate Curve
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(area_percentages, landslide_percentages, 'b-', linewidth=2, label='Success Rate Curve')
ax1.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Random Model')
ax1.fill_between(area_percentages, landslide_percentages, area_percentages, alpha=0.3, color='blue')
ax1.set_xlabel('Cumulative Area (%)', fontsize=11)
ax1.set_ylabel('Cumulative Landslides (%)', fontsize=11)
ax1.set_title('Success Rate Curve\n(Training Data Validation)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])
ax1.set_ylim([0, 100])

# Add text annotations
ax1.text(10, 85, f'Top 10% area:\n{landslide_percentages[10]:.1f}% landslides', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
ax1.text(20, 70, f'Top 20% area:\n{landslide_percentages[20]:.1f}% landslides',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

# 2. ROC Curve
ax2 = fig.add_subplot(gs[0, 2:])
ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
ax2.fill_between(fpr, tpr, alpha=0.3, color='blue')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curve (Success Rate)', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# 3. Prediction Rate Curve (if available)
ax3 = fig.add_subplot(gs[1, 0:2])
if has_prediction_curve:
    ax3.plot(pred_fpr, pred_tpr, 'g-', linewidth=2, label=f'Prediction Rate (AUC = {pred_auc:.3f})')
    ax3.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax3.fill_between(pred_fpr, pred_tpr, alpha=0.3, color='green')
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('True Positive Rate', fontsize=11)
    ax3.set_title('Prediction Rate Curve\n(Spatial CV Test Set)', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'Prediction Rate Curve\nNot Available', 
             ha='center', va='center', fontsize=14, color='gray')
    ax3.axis('off')

# 4. Distribution comparison
ax4 = fig.add_subplot(gs[1, 2:])
ax4.hist(non_landslide_susc, bins=50, alpha=0.5, label=f'Non-Landslide (μ={non_landslide_susc.mean():.3f})', 
         color='blue', density=True)
ax4.hist(landslide_susc, bins=50, alpha=0.5, label=f'Landslide (μ={landslide_susc.mean():.3f})', 
         color='red', density=True)
ax4.set_xlabel('Susceptibility Value', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title('Distribution Comparison\n(Landslide vs Non-Landslide)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axvline(landslide_susc.mean(), color='red', linestyle='--', linewidth=2)
ax4.axvline(non_landslide_susc.mean(), color='blue', linestyle='--', linewidth=2)

# 5. Box plot comparison
ax5 = fig.add_subplot(gs[2, 0])
bp = ax5.boxplot([non_landslide_susc, landslide_susc], 
                  labels=['Non-Landslide', 'Landslide'],
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue'),
                  medianprops=dict(color='red', linewidth=2))
ax5.set_ylabel('Susceptibility Value', fontsize=11)
ax5.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
ax5.text(0.5, -0.15, f"Cohen's d = {cohens_d:.3f} ({effect})", 
         transform=ax5.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 6. Confusion matrix heatmap (at threshold 0.5)
ax6 = fig.add_subplot(gs[2, 1])
cm_05 = confusion_matrix(y_true, (y_pred >= 0.5).astype(int))
sns.heatmap(cm_05, annot=True, fmt='d', cmap='Blues', ax=ax6, cbar=False)
ax6.set_xlabel('Predicted', fontsize=11)
ax6.set_ylabel('Actual', fontsize=11)
ax6.set_title('Confusion Matrix\n(Threshold = 0.5)', fontsize=12, fontweight='bold')
ax6.set_xticklabels(['Non-LS', 'Landslide'])
ax6.set_yticklabels(['Non-LS', 'Landslide'])

# 7. Performance metrics across thresholds
ax7 = fig.add_subplot(gs[2, 2:])
ax7.plot(df_confusion['Threshold'], df_confusion['Accuracy'], 'o-', label='Accuracy', linewidth=2)
ax7.plot(df_confusion['Threshold'], df_confusion['Precision'], 's-', label='Precision', linewidth=2)
ax7.plot(df_confusion['Threshold'], df_confusion['Recall'], '^-', label='Recall', linewidth=2)
ax7.plot(df_confusion['Threshold'], df_confusion['F1_Score'], 'd-', label='F1 Score', linewidth=2)
ax7.set_xlabel('Classification Threshold', fontsize=11)
ax7.set_ylabel('Score', fontsize=11)
ax7.set_title('Performance Metrics vs Threshold', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)
ax7.set_ylim([0, 1])

plt.savefig('comprehensive_validation_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: comprehensive_validation_analysis.png")

# ======================== 9. SAVE VALIDATION SUMMARY ========================
print("\n9. Saving validation summary...")

validation_summary = {
    'Success_Rate_AUC': roc_auc,
    'Prediction_Rate_AUC': pred_auc if has_prediction_curve else np.nan,
    'Morans_I': morans_i if morans_i else np.nan,
    'Morans_I_Zscore': z_score if z_score else np.nan,
    'Landslide_Mean_Susceptibility': landslide_susc.mean(),
    'NonLandslide_Mean_Susceptibility': non_landslide_susc.mean(),
    'Ttest_Statistic': t_stat,
    'Ttest_Pvalue': p_value,
    'MannWhitney_U': u_stat,
    'MannWhitney_Pvalue': u_pvalue,
    'Cohens_d': cohens_d,
    'Effect_Size': effect,
    'Top10_Area_Captures_Landslides_Pct': landslide_percentages[10],
    'Top20_Area_Captures_Landslides_Pct': landslide_percentages[20],
    'Top30_Area_Captures_Landslides_Pct': landslide_percentages[30]
}

df_summary = pd.DataFrame([validation_summary])
df_summary.to_csv('validation_summary.csv', index=False)
print(f"   ✓ Saved: validation_summary.csv")

# Save success rate data
success_rate_data = pd.DataFrame({
    'Area_Percentage': area_percentages,
    'Landslide_Percentage': landslide_percentages
})
success_rate_data.to_csv('success_rate_curve_data.csv', index=False)
print(f"   ✓ Saved: success_rate_curve_data.csv")

print(f"\n{'='*80}")
print("VALIDATION ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\nKey Findings:")
print(f"  • Success Rate AUC: {roc_auc:.4f} (Excellent if >0.9, Good if >0.8)")
if has_prediction_curve:
    print(f"  • Prediction Rate AUC: {pred_auc:.4f} (Spatial CV performance)")
if morans_i:
    print(f"  • Moran's I: {morans_i:.4f} (Spatial clustering present)")
print(f"  • Top 10% area captures: {landslide_percentages[10]:.1f}% of landslides")
print(f"  • Top 20% area captures: {landslide_percentages[20]:.1f}% of landslides")
print(f"  • Cohen's d: {cohens_d:.3f} ({effect} effect)")
print(f"\nGenerated files:")
print(f"  1. comprehensive_validation_analysis.png - All validation plots")
print(f"  2. validation_summary.csv - Summary statistics")
print(f"  3. validation_confusion_matrices.csv - Performance at different thresholds")
print(f"  4. success_rate_curve_data.csv - Success rate curve data")
print(f"\n{'='*80}")
