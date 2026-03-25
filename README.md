# Landslide Susceptibility Modelling

This repository contains a leakage-aware landslide susceptibility modelling workflow built around:

- an artificial neural network (ANN)
- benchmark comparisons with Random Forest (RF) and Gradient Boosting (GB)
- spatially explicit evaluation using 5-fold block cross-validation
- raster-based susceptibility map generation from trained model outputs

The current code is designed for academic comparison of models under the same sampling setup, predictor family, and spatial validation logic.

## Overview

The project uses two raw point datasets:

- landslide points
- non-landslide points

Each point contains:

- 12 continuous predictors: `aspect`, `elv`, `flowAcc`, `planCurv`, `profCurv`, `riverProx`, `roadProx`, `slope`, `SPI`, `TPI`, `TRI`, `TWI`
- 2 categorical predictors: `lithology`, `soil`
- coordinates: `xcoord`, `ycoord`
- optional identifier: `fid`

## Repository Layout

```text
ANN-landslide-susceptibility/
├── data/
│   └── processed/
│       ├── landslides.csv
│       ├── non_landslides.csv
│       ├── output_landslides.csv
│       └── output_non_landslides.csv
├── alignedRaster/                  # or alignedRasters/
│   ├── aspect_utm15_aligned.tif
│   ├── elv_aligned.tif
│   ├── ...
│   ├── lithology_aligned.tif
│   └── soil_aligned.tif
├── models/
│   ├── best_model_advanced.pth
│   └── landslide_model_advanced_complete.pth
├── results/
│   ├── benchmark/
│   ├── susceptibility_maps/
│   ├── training/
│   ├── transfer_learning/
│   └── validation/
├── benchmark_spatial_models.py
├── comprehensive_validation.py
├── modelTraining.py
├── preprocessing.py
├── project_paths.py
├── train.py
└── transfer_learning_durban.py
```

## Required Input Files

### Raw training point tables

Place the original point tables in one of the supported locations, preferably:

- `data/processed/landslides.csv`
- `data/processed/non_landslides.csv`

These raw files should contain the unnormalized continuous predictors and integer `lithology` and `soil` classes.

### Raster stack for map prediction

Place the aligned rasters in `alignedRaster/` using these names. The prediction script also checks `alignedRasters/` and several parent-folder fallback locations, but `alignedRaster/` in the project root is the clearest default.

- `aspect_utm15_aligned.tif`
- `elv_aligned.tif`
- `flow_acc_aligned.tif`
- `planCurv_aligned.tif`
- `profCurv_aligned.tif`
- `riversprox_aligned.tif`
- `roadsprox_aligned.tif`
- `slope_aligned.tif`
- `SPI_aligned.tif`
- `TPI_aligned.tif`
- `TRI_aligned.tif`
- `TWI_aligned.tif`
- `lithology_aligned.tif`
- `soil_aligned.tif`

## Environment Setup

Create and activate a virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn torch matplotlib seaborn rasterio scipy
```

If you already use Conda or another environment manager, that is also fine as long as the packages above are available.

For the optional Durban transfer-learning workflow, you may also need:

```bash
pip install geopandas fiona
```

## How To Run

### 1. Train the ANN

```bash
python modelTraining.py
```

This produces:

- model package: `models/landslide_model_advanced_complete.pth`
- best checkpoint: `models/best_model_advanced.pth`
- metrics and plots in `results/training/`

Main outputs include:

- `results/training/training_summary.json`
- `results/training/final_test_metrics.csv`
- `results/training/selected_feature_importance.csv`
- `advanced_evaluation.png`
- `advanced_validation_evaluation.png`
- `feature_importance_advanced.png`
- `training_analysis.png`

### 2. Run the spatial benchmark comparison

```bash
python benchmark_spatial_models.py
```

This runs ANN, RF, and GB under the same 5-fold spatial block cross-validation framework and writes outputs to `results/benchmark/`.

Main outputs include:

- `results/benchmark/benchmark_results_detailed.csv`
- `results/benchmark/benchmark_results_summary.csv`
- `results/benchmark/benchmark_fold_assignments.csv`
- `results/benchmark/benchmark_fold_features.csv`
- `results/benchmark/benchmark_metadata.json`

### 3. Generate the susceptibility map

```bash
python train.py
```

This loads the saved ANN model package and applies it to the aligned raster stack. The output map is written to:

- `results/susceptibility_maps/susceptibility_map.tif`

### 4. Optional transfer learning experiment

```bash
python transfer_learning_durban.py
```

This evaluates the trained Chiapas model on the Durban dataset and writes outputs to `results/transfer_learning/`.

### 5. Optional susceptibility map validation

```bash
python comprehensive_validation.py
```

This validates the generated susceptibility map against known landslide and non-landslide locations and writes outputs to `results/validation/`.

## Detailed Pipeline

## 1. Raw Data Ingestion

`modelTraining.py` and `benchmark_spatial_models.py` both load the raw point tables, not the pre-normalized exports.

The scripts:

- read landslide and non-landslide CSVs
- add `label = 1` for landslides and `label = 0` for non-landslides
- concatenate both classes into one modelling table
- convert booleans to numeric if needed
- coerce invalid numeric values
- preserve `xcoord` and `ycoord` for spatial blocking

## 2. Spatial Blocking

Spatial evaluation is based on a 5×5 block grid.

The helper function in both training and benchmarking:

- computes evenly spaced bins in X and Y
- assigns each point to one of 25 spatial blocks
- removes very small blocks using `MIN_BLOCK_SIZE`

The exact number of populated blocks retained for modelling depends on the current dataset after `MIN_BLOCK_SIZE` filtering.

## 3. Leakage-Safe Preprocessing

The shared preprocessing logic lives in `preprocessing.py`.

For each training split or outer fold:

1. the preprocessor is fit on the training data only
2. continuous variables are min-max scaled using training-only minima and maxima
3. categorical variables are one-hot encoded using training-only observed classes
4. validation/test data are transformed using that fitted training preprocessor

This means:

- no global scaling is fit before splitting
- no test-fold information is used to define continuous scaling
- no test-fold-only lithology or soil categories influence the training design matrix

## 4. Feature Selection

Feature selection is performed using training data only.

The selection function combines:

- `SelectKBest` with ANOVA F-statistics
- Random Forest feature importance
- Recursive Feature Elimination (RFE)

The final ranking is based on ensemble voting and importance ordering, and the top `MAX_SELECTED_FEATURES` features are retained.

In the benchmark script, this feature selection is repeated separately inside each outer fold to keep the comparison leakage-safe.

## 5. Final Scaling Before ANN

After feature engineering and feature selection:

- the ANN pipeline fits a `RobustScaler` on the selected training features only
- validation and test subsets are transformed using that same fitted scaler

This makes the ANN less sensitive to outliers after the initial training-only min-max stage.

## 6. ANN Training Workflow

The ANN is defined in `modelTraining.py` as `ImprovedLandslideANN`.

Training uses:

- binary cross-entropy with logits
- class weighting via `pos_weight`
- `AdamW` optimizer
- `ReduceLROnPlateau` scheduler
- gradient clipping
- early stopping

The workflow is:

1. create spatial blocks
2. split into spatial `train+validation` and `test`
3. split the `train+validation` subset again into `train` and `validation`
4. fit preprocessing on training only
5. select features on training only
6. fit `RobustScaler` on training only
7. train the ANN using the training subset
8. save the best checkpoint by validation loss
9. tune the classification threshold on the validation probabilities
10. report final metrics on validation and test

## 7. Benchmark Workflow

The benchmark comparison in `benchmark_spatial_models.py` is designed to compare ANN, RF, and GB under the same spatial framework.

For each outer fold:

1. the test fold is held out spatially
2. preprocessing is fit on the outer training fold only
3. feature selection is fit on the outer training fold only
4. ANN, RF, and GB are trained on the same selected predictor set
5. each model tunes its decision threshold on an inner training/validation split drawn only from the outer training fold
6. the tuned model is evaluated on the outer test fold

Reported metrics include:

- AUROC
- PR-AUC
- Accuracy
- Precision
- Recall
- F1
- MCC
- Brier score
- selected threshold

This benchmark design is the main evidence base for manuscript model comparison.

## 8. Model Package Contents

The saved ANN model package includes:

- `model_state_dict`
- selected feature list
- trained `RobustScaler`
- saved training preprocessor
- best validation threshold
- metadata about training metrics and package versions

This allows `train.py` to apply the same learned preprocessing logic during raster inference.

## 9. Raster Prediction Workflow

`train.py` performs map prediction as follows:

1. locate the aligned raster stack
2. confirm all rasters have matching dimensions
3. read the saved ANN package
4. extract the saved training preprocessor and `RobustScaler`
5. process the raster stack in windows to control memory use
6. scale the continuous raster bands using the saved training-only minima and maxima
7. one-hot encode `lithology` and `soil` using the saved training categories
8. subset to the selected features used by the ANN
9. apply the saved `RobustScaler`
10. run ANN inference and write probabilities to the output raster

The previous hand-tuned edge correction has been removed so the susceptibility map is now the direct model output.

## Outputs and Interpretation

### Training outputs

The ANN training summary is stored in:

- `results/training/training_summary.json`

### Benchmark outputs

The model comparison summary is stored in:

- `results/benchmark/benchmark_results_summary.csv`

This is the main file to cite when comparing ANN, RF, and GB for the manuscript.

### Susceptibility map

The raster probability map is stored in:

- `results/susceptibility_maps/susceptibility_map.tif`

## Reproducibility Notes

- random seeds are fixed in the training and benchmark scripts
- preprocessing is fit inside the training data only
- benchmark feature selection is fold-specific
- threshold tuning is done on validation data, not on the held-out test data
- the model package stores the preprocessor needed for consistent raster inference

## Recommended Execution Order

For a full end-to-end run:

```bash
python modelTraining.py
python benchmark_spatial_models.py
python train.py
```

Optional:

```bash
python transfer_learning_durban.py
```

Optional after map generation:

```bash
python comprehensive_validation.py
```

