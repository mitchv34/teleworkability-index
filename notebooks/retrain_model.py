import marimo

__generated_with = "0.12.5"
app = marimo.App(width="full", app_title="Retrain Teleworkability Model")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import estimation code
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from wfh_share_estimation import DataStore, ModelPipeline
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    return (
        DataStore,
        ModelPipeline,
        Path,
        RandomForestClassifier,
        RandomForestRegressor,
        mo,
        np,
        pd,
        sys,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        # 🧪 Retrain Teleworkability Model
        
        **Adjust hyperparameters and retrain the two-stage Random Forest model**
        
        This notebook allows you to experiment with different model parameters and see how they affect 
        the teleworkability predictions. Training takes approximately 10-30 seconds depending on 
        the number of trees selected.
        
        ⚠️ **Note:** This performs actual model training in your browser using WebAssembly. 
        The computation may take some time, especially with higher tree counts.
        
        ---
        """
    )
    return


@app.cell
def __(mo):
    mo.md("## Model Hyperparameters")
    
    # Classifier parameters
    n_estimators_clf = mo.ui.slider(
        start=10, 
        stop=200, 
        value=50, 
        step=10,
        label="Classifier: Number of Trees",
        show_value=True
    )
    
    max_depth_clf = mo.ui.slider(
        start=5,
        stop=50,
        value=20,
        step=5,
        label="Classifier: Max Depth",
        show_value=True
    )
    
    # Regressor parameters  
    n_estimators_reg = mo.ui.slider(
        start=10,
        stop=200,
        value=80,
        step=10,
        label="Regressor: Number of Trees",
        show_value=True
    )
    
    max_depth_reg = mo.ui.slider(
        start=5,
        stop=50,
        value=20,
        step=5,
        label="Regressor: Max Depth",
        show_value=True
    )
    
    # Other parameters
    zero_threshold = mo.ui.slider(
        start=0.5,
        stop=0.95,
        value=0.8,
        step=0.05,
        label="Zero Probability Threshold",
        show_value=True
    )
    
    test_size = mo.ui.slider(
        start=0.1,
        stop=0.4,
        value=0.2,
        step=0.05,
        label="Test Set Size",
        show_value=True
    )
    
    mo.vstack([
        mo.md("**Classifier (Stage 1: Detect Zero/Non-Zero)**"),
        mo.hstack([n_estimators_clf, max_depth_clf]),
        mo.md("**Regressor (Stage 2: Predict Teleworkability)**"),
        mo.hstack([n_estimators_reg, max_depth_reg]),
        mo.md("**Other Parameters**"),
        mo.hstack([zero_threshold, test_size])
    ])
    return (
        max_depth_clf,
        max_depth_reg,
        n_estimators_clf,
        n_estimators_reg,
        test_size,
        zero_threshold,
    )


@app.cell
def __(mo):
    # Train button
    train_button = mo.ui.button(
        label="🚀 Train Model",
        kind="success",
        disabled=False
    )
    
    mo.md("## Train Model")
    train_button
    return (train_button,)


@app.cell
def __(
    DataStore,
    ModelPipeline,
    Path,
    RandomForestClassifier,
    RandomForestRegressor,
    max_depth_clf,
    max_depth_reg,
    mo,
    n_estimators_clf,
    n_estimators_reg,
    test_size,
    train_button,
    zero_threshold,
):
    # Training logic
    results = None
    new_metrics = None
    new_predictions = None
    training_complete = False
    
    if train_button.value:
        mo.output.clear()
        mo.md("### Training in progress... ⏳")
        
        try:
            # Load data
            DATA_LIST = ['WORK_CONTEXT', 'WORK_ACTIVITIES', 'SKILLS', 'ABILITIES']
            
            ds = DataStore(
                data_list=DATA_LIST,
                metric='importance',
                data_dir_ors=str(Path(__file__).parent.parent / "data" / "proc" / "ors"),
                data_dir_onet=str(Path(__file__).parent.parent / "data" / "onet_data" / "processed" / "measure"),
                data_dir_onet_reference=str(Path(__file__).parent.parent / "data" / "onet_data" / "processed" / "reference"),
                test_size=test_size.value
            )
            
            # Create models with user parameters
            clf_model = RandomForestClassifier(
                n_estimators=n_estimators_clf.value,
                max_depth=max_depth_clf.value if max_depth_clf.value < 50 else None,
                random_state=42
            )
            
            reg_model = RandomForestRegressor(
                n_estimators=n_estimators_reg.value,
                max_depth=max_depth_reg.value if max_depth_reg.value < 50 else None,
                random_state=42
            )
            
            # Train pipeline
            mp = ModelPipeline(
                data=ds,
                classifier_model=clf_model,
                regressor_model=reg_model,
                normalize='logit',
                zero_threshold=zero_threshold.value,
                suppress_messages=True
            )
            
            mp.train()
            
            # Evaluate
            train_results = mp.evaluate(split='train', verbose=False)
            test_results = mp.evaluate(split='test', verbose=False)
            
            # Store results
            new_predictions = mp.predict_unlabeled()
            new_metrics = {
                'train': mp.scores.copy(),
                'split': 'train'
            }
            
            # Evaluate on test
            _ = mp.evaluate(split='test', verbose=False)
            test_metrics = mp.scores.copy()
            test_metrics['split'] = 'test'
            
            training_complete = True
            
        except Exception as e:
            mo.md(f"❌ **Training failed:** {str(e)}")
    
    return (
        DATA_LIST,
        clf_model,
        ds,
        mp,
        new_metrics,
        new_predictions,
        reg_model,
        results,
        test_metrics,
        test_results,
        train_results,
        training_complete,
    )


@app.cell
def __(mo, new_metrics, pd, test_metrics, training_complete):
    if training_complete:
        mo.md("### ✅ Training Complete!")
        
        # Create metrics comparison
        metrics_df = pd.DataFrame([new_metrics, test_metrics])
        
        mo.md("**Model Performance:**")
        mo.ui.table(metrics_df)
    return (metrics_df,)


@app.cell
def __(Path, mo, pd, training_complete):
    if training_complete:
        # Load baseline for comparison
        baseline_path = Path(__file__).parent.parent / "results" / "model_metrics.csv"
        baseline = pd.read_csv(baseline_path)
        
        mo.md("### 📊 Comparison with Baseline")
        mo.md("**Baseline Model Performance (from saved results):**")
        mo.ui.table(baseline)
    return baseline, baseline_path


@app.cell
def __(mo, new_predictions, training_complete):
    if training_complete and new_predictions is not None:
        mo.md(f"### 📥 Download New Predictions")
        mo.md(f"Total predictions: {len(new_predictions):,} occupations")
        
        # Prepare download
        csv_data = new_predictions.to_csv()
        
        mo.download(
            data=csv_data.encode(),
            filename="teleworkability_retrained.csv",
            mimetype="text/csv",
            label="Download Predictions CSV"
        )
    return csv_data,


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        
        ## About the Model
        
        **Two-Stage Random Forest Pipeline:**
        
        1. **Stage 1 (Classifier):** Predicts whether an occupation can be done remotely at all (zero vs non-zero)
           - Uses all O*NET features (importance ratings)
           - Binary classification with calibrated probabilities
           
        2. **Stage 2 (Regressor):** Predicts the teleworkability fraction for non-zero occupations
           - Only trained on occupations classified as non-zero
           - Predicts values in [0, 1] range
           - Uses logit transformation for better performance
        
        **Key Hyperparameters:**
        
        - **Number of Trees**: More trees → better accuracy but slower training (50-200 recommended)
        - **Max Depth**: Limits tree depth to prevent overfitting (10-30 typical, None = unlimited)
        - **Zero Threshold**: Probability threshold for classifying as "can be done remotely" (0.7-0.9 typical)
        - **Test Size**: Fraction of labeled data held out for evaluation (0.2 = 20%)
        
        **Feature Sources:**
        
        - Work Context (e.g., "Face-to-Face Discussions", "Physical Proximity")
        - Work Activities (e.g., "Analyzing Data", "Repairing Equipment")
        - Skills (e.g., "Programming", "Active Listening")
        - Abilities (e.g., "Oral Expression", "Manual Dexterity")
        
        **Citation:** Valdes-Bobes, M. & Lukianova, A. (2025). "Why Remote Work Stuck: A Structural Decomposition"
        """
    )
    return


if __name__ == "__main__":
    app.run()
