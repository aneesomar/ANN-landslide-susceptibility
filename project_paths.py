from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_RESULTS_DIR = RESULTS_DIR / "training"
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmark"
VALIDATION_RESULTS_DIR = RESULTS_DIR / "validation"
TRANSFER_RESULTS_DIR = RESULTS_DIR / "transfer_learning"
SUSCEPTIBILITY_MAPS_DIR = RESULTS_DIR / "susceptibility_maps"


def ensure_project_dirs():
    for directory in [
        DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        TRAINING_RESULTS_DIR,
        BENCHMARK_RESULTS_DIR,
        VALIDATION_RESULTS_DIR,
        TRANSFER_RESULTS_DIR,
        SUSCEPTIBILITY_MAPS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_raw_csvs():
    candidates = [
        (
            PROCESSED_DATA_DIR / "landslides.csv",
            PROCESSED_DATA_DIR / "non_landslides.csv",
        ),
        (
            DATA_DIR / "landslides.csv",
            DATA_DIR / "non_landslides.csv",
        ),
        (
            PROJECT_ROOT / "landslides.csv",
            PROJECT_ROOT / "non_landslides.csv",
        ),
    ]
    for landslide_path, non_landslide_path in candidates:
        if landslide_path.exists() and non_landslide_path.exists():
            return landslide_path, non_landslide_path
    raise FileNotFoundError("Could not find raw landslides.csv and non_landslides.csv")


def resolve_processed_csvs():
    candidates = [
        (
            PROCESSED_DATA_DIR / "output_landslides.csv",
            PROCESSED_DATA_DIR / "output_non_landslides.csv",
        ),
        (
            PROCESSED_DATA_DIR / "landslides.csv",
            PROCESSED_DATA_DIR / "non_landslides.csv",
        ),
        (
            PROJECT_ROOT / "output_landslides.csv",
            PROJECT_ROOT / "output_non_landslides.csv",
        ),
        (
            PROJECT_ROOT.parent / "output_landslides.csv",
            PROJECT_ROOT.parent / "output_non_landslides.csv",
        ),
    ]
    for landslide_path, non_landslide_path in candidates:
        if landslide_path.exists() and non_landslide_path.exists():
            return landslide_path, non_landslide_path
    raise FileNotFoundError("Could not find output_landslides.csv and output_non_landslides.csv")


def resolve_processed_landslide_csv():
    landslide_path, _ = resolve_processed_csvs()
    return landslide_path


def resolve_model_package():
    candidates = [
        MODELS_DIR / "landslide_model_advanced_complete.pth",
        PROJECT_ROOT / "landslide_model_advanced_complete.pth",
    ]
    for model_path in candidates:
        if model_path.exists():
            return model_path
    raise FileNotFoundError("Could not find landslide_model_advanced_complete.pth")


def resolve_susceptibility_map():
    candidates = [
        SUSCEPTIBILITY_MAPS_DIR / "susceptibility_map.tif",
        PROJECT_ROOT / "susceptibility_map.tif",
    ]
    for map_path in candidates:
        if map_path.exists():
            return map_path
    raise FileNotFoundError("Could not find susceptibility_map.tif")
