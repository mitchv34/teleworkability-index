# Minimal runner for ModelPipeline
import os, sys
sys.path.append(os.path.dirname(__file__))
from wfh_share_estimation import DataStore, ModelPipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
from pathlib import Path
import json

def _build_estimator_from_params(default_cls, params):
    """Try to instantiate an sklearn estimator from params dict.

    If instantiation fails, return None so caller can fall back to a default.
    """
    if not params:
        return None
    # If params is already an estimator object, return it (best_models may sometimes store this)
    try:
        # sklearn estimators expose get_params; duck-typing check
        if hasattr(params, 'get_params'):
            return params
    except Exception:
        pass

    if isinstance(params, dict):
        try:
            return default_cls(**params)
        except Exception:
            # Could not instantiate (type mismatch or unexpected keys). Return None to fallback.
            return None
    # Unknown type
    return None

# Choose a subset of ONET measure sources present in data/onet_data/processed/measure
DATA_LIST = [
    'WORK_CONTEXT',
    'WORK_ACTIVITIES',
    'SKILLS',
    'ABILITIES',
]

def main():
    ds = DataStore(
        data_list=DATA_LIST,
        metric='importance',
        data_dir_ors='data/proc/ors/',
        data_dir_onet='data/onet_data/processed/measure/',
        data_dir_onet_reference='data/onet_data/processed/reference/'
    )

    mp = ModelPipeline(
        data=ds,
        # Reduced estimator counts for a fast smoke-test run. Increase for final training.
        classifier_model=RandomForestClassifier(n_estimators=50, random_state=42),
        regressor_model=RandomForestRegressor(n_estimators=80, random_state=42),
        normalize='logit',
        zero_threshold=0.8,
        suppress_messages=False,
    )
    mp.train()
    # Evaluate on training data
    train_res = mp.evaluate(split='train', verbose=True)
    print("\nHead of TRAIN results (actual vs predicted):")
    print(train_res.head())

    # Save outputs (minimal) and full export
    outdir = Path('results/wfh_minimal')
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([mp.scores]).to_csv(outdir / 'train_metrics.csv', index=False)
    train_res.to_csv(outdir / 'train_predictions.csv')
    # Full export to repo results dir
    mp.export_all_results(out_dir='results')

    # Predict unlabeled records (if any)
    if not ds.unlabeled_data.empty:
        preds = mp.predict_unlabeled()
        print(f"\nPredicted {len(preds)} unlabeled rows.")

if __name__ == '__main__':
    main()

