import pandas as pd

from project_paths import PROCESSED_DATA_DIR, ensure_project_dirs, resolve_raw_csvs


def main():
    ensure_project_dirs()
    landslide_path, non_landslide_path = resolve_raw_csvs()
    landslides = pd.read_csv(landslide_path)
    non_landslides = pd.read_csv(non_landslide_path)

    # This script is now a legacy export helper only.
    # Model training and benchmarking perform scaling inside each training fold.
    landslides_export = pd.get_dummies(landslides, columns=["lithology", "soil"])
    non_landslides_export = pd.get_dummies(non_landslides, columns=["lithology", "soil"])

    landslide_output = PROCESSED_DATA_DIR / "output_landslides.csv"
    non_landslide_output = PROCESSED_DATA_DIR / "output_non_landslides.csv"
    landslides_export.to_csv(landslide_output, index=False)
    non_landslides_export.to_csv(non_landslide_output, index=False)

    print("Saved legacy one-hot encoded exports for inspection only:")
    print(f"  {landslide_output}")
    print(f"  {non_landslide_output}")
    print("No normalization was applied here. Training and benchmarking now scale inside each training split.")


if __name__ == "__main__":
    main()
