import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter



##########################data preparation##########################

# Load datasets
landslides = pd.read_csv("output_landslides.csv")
nonLandslides = pd.read_csv("output_non_landslides.csv")

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

# Drop unwanted columns BEFORE splitting
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

# Train-test split (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# Use RobustScaler instead of StandardScaler (better for outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames to maintain column names for saving
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


#save the split datasets
X_train_scaled_df.to_csv("X_train.csv", index=False)
X_test_scaled_df.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

############################ Convert to torch tensors ############################
# Convert to torch tensors using scaled data
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Calculate class weights for imbalanced learning
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Create weighted sampler for balanced training
y_train_np = y_train.values
sample_weights = [class_weight_dict[label] for label in y_train_np]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# Wrap into datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Dataloaders with weighted sampling
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)  # Increased batch size
test_loader = DataLoader(test_dataset, batch_size=64)



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

model = AdvancedLandslideANN(X_train_scaled.shape[1])

# Use focal loss for better handling of difficult examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

# Advanced learning rate scheduling with warm restart
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

# Advanced training loop with gradient clipping and mixed precision
num_epochs = 150
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 25
patience_counter = 0

# Enable mixed precision training for faster computation
scaler_amp = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Training on device: {device}")

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
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
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
    avg_val_loss = val_loss / len(test_loader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    scheduler.step()
    
    if epoch % 15 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Early stopping with patience
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model_advanced.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
# Load best model
model.load_state_dict(torch.load('best_model_advanced.pth'))

# Move tensors to device for evaluation
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Test different thresholds for optimal performance
thresholds = np.arange(0.3, 0.8, 0.05)
best_threshold = 0.5
best_f1 = 0

model.eval()
with torch.no_grad():
    all_outputs = model(X_test_tensor)
    all_probabilities = torch.sigmoid(all_outputs).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

f1_scores = []
for threshold in thresholds:
    predictions = (all_probabilities > threshold).astype(int)
    f1 = f1_score(y_true, predictions)
    f1_scores.append(f1)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

plt.subplot(1, 2, 2)
plt.plot(thresholds, f1_scores, 'b-o')
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best threshold: {best_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Best threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")

def evaluate_model_advanced(model, X_test_tensor, y_test_tensor, threshold=0.5, device='cpu'):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > threshold).int().cpu().numpy()
        true = y_test_tensor.int().cpu().numpy()
        probs_np = probabilities.cpu().numpy()

        acc = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, zero_division=0)
        recall = recall_score(true, predicted, zero_division=0)
        f1 = f1_score(true, predicted, zero_division=0)
        auc = roc_auc_score(true, probs_np)

        print(f"\nAdvanced Evaluation (threshold={threshold}):")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(true, predicted, target_names=['Non-Landslide', 'Landslide']))
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(true, probs_np)
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Confusion matrix
        cm = confusion_matrix(true, predicted)
        plt.subplot(1, 3, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Non-Landslide', 'Landslide'],
                    yticklabels=['Non-Landslide', 'Landslide'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Prediction probability distribution
        plt.subplot(1, 3, 3)
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
        
        return acc, precision, recall, f1, auc

# Evaluate with default threshold
print("=== Evaluation with default threshold (0.5) ===")        
evaluate_model_advanced(model, X_test_tensor, y_test_tensor, threshold=0.5, device=device)

# Evaluate with optimized threshold
print(f"\n=== Evaluation with optimized threshold ({best_threshold}) ===")
evaluate_model_advanced(model, X_test_tensor, y_test_tensor, threshold=best_threshold, device=device)

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



