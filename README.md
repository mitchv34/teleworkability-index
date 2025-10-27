Teleworkability Index (ψ$\) - Minimal, Self-Contained Repository

This repository contains a minimal, self-contained implementation to construct the occupation-level teleworkability index ψ from O*NET features and ORS labels.

Contents
- wfh_share_estimation.py: Core pipeline (DataLoader, DataStore, Mode
- run_wfh_share_minimal.py: Minimal entrypoint to train, evaluate, and export results
- data/
  - proc/ors/final_second_wave_2023.csv (labels)
  - onet_data/processed/measure/*.csv (features)
  - onet_data/processed/reference/*.csv (scale and content reference)
- results/ (outputs written here)
- requirements.txt

Quick start
1) Create a Python environment and install dependencies:
   pip install -r requirements.txt

2) Run the minimal pipeline:
   python run_wfh_share_minimal.py

Outputs
- results/wfh_minimal/train_metrics.csv
- results/wfh_minimal/train_predictions.csv
- results/full_occupation_predictions.csv
- results/model_metrics.csv
- results/classifier_feature_importance.csv
- results/regressor_feature_importance.csv

Notes
- The code uses repo-relative data paths (data/...). No external data is required beyond what is included here.
- This bundle excludes grid-search orchestration, Slurm artifacts, and bootstrap infrastructure from the main project.
