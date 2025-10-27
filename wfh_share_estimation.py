# %%
"""
Title: 01_wfh_share_estimation.py
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2025-02-12
Description: Minimal two-stage pipeline for WFH share estimation.
Notes:
- Uses ELEMENT_ID-based features only (no readability mapping).
- Removes plotting and extra helper classes for simplicity.
- Fixes metric selection, ORS scaling, and prediction robustness.
"""

# Importing necessary libraries
import os
import re
import numpy as np
import pandas as pd


# Importing machine learning libraries from scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


#*=========================================================================================
#* DEFAULTS AND CONSTANTS
#*=========================================================================================

DATA_DIR_ORS = 'data/proc/ors/'
DATA_DIR_ONET = 'data/onet_data/processed/measure/'
# Default to repo-relative reference path; allow override via constructor.
DATA_DIR_ONET_REFERENCE = 'data/onet_data/processed/reference/'

# %%
#?=========================================================================================
#? DATA LOADER
#?=========================================================================================
class DataLoader:
    """
    Loads both the ONET and ORS datasets.
    """
    def __init__(self,
                data_dir_ors = DATA_DIR_ORS, 
                data_dir_onet = DATA_DIR_ONET, 
                data_dir_onet_reference = DATA_DIR_ONET_REFERENCE):
        """
        Initializes the DataLoader with the specified directories.
        
        Parameters:
        data_dir_ors (str): Directory path for ORS data. (Contains labels)
        data_dir_onet (str): Directory path for ONET data. (Contains features)
        data_dir_onet_reference (str): Directory path for ONET reference data. (Contains category descriptions)
        """

        
        self.data_dir_ors = data_dir_ors
        
        self.data_dir_onet = data_dir_onet
        self.data_dir_onet_reference = data_dir_onet_reference

    def load_onet_data(self, data_list):
        """Load and pivot ONET data from multiple sources using ELEMENT_ID and SCALE_NAME.

        Returns a DataFrame indexed by ONET_SOC_CODE with MultiIndex columns
        (ELEMENT_ID, SCALE_NAME).
        """
        # Load scale reference file with fallback to SCALE_ID if missing.
        scale_ref_path = os.path.join(self.data_dir_onet_reference, 'SCALES_REFERENCE.csv')
        scale_reference = None
        if os.path.exists(scale_ref_path):
            scale_reference = (
                pd.read_csv(scale_ref_path)
                .set_index('SCALE_ID')['SCALE_NAME']
                .to_dict()
            )
        scales_to_exclude = {'IH', 'VH'}

        df = pd.DataFrame()
        for data_source in data_list:
            data_path = os.path.join(self.data_dir_onet, f'{data_source}.csv')
            data = pd.read_csv(data_path)

            # Build SCALE_NAME, mapping SCALE_ID to readable names if possible.
            if scale_reference is not None:
                data['SCALE_NAME'] = data['SCALE_ID'].map(scale_reference).apply(
                    lambda x: re.sub(r"\(.*\)", "", x).strip().replace(" ", "_").lower()
                )
            else:
                data['SCALE_NAME'] = data['SCALE_ID']

            # Drop excluded scales if present.
            if 'SCALE_ID' in data.columns:
                data = data[~data['SCALE_ID'].isin(scales_to_exclude)]

            # If CATEGORY exists, append to SCALE_NAME (e.g., 'importance_1').
            if 'CATEGORY' in data.columns:
                data['CATEGORY'] = data['CATEGORY'].fillna("")
                data['CATEGORY'] = data['CATEGORY'].apply(lambda x: str(int(x)) if x != "" else x)
                data['SCALE_NAME'] = data['SCALE_NAME'] + data['CATEGORY'].apply(lambda x: "_" if len(x) > 0 else "") + data['CATEGORY']

            # Pivot on ELEMENT_ID and SCALE_NAME only.
            columns_pivot = ['ELEMENT_ID', 'SCALE_NAME']
            data = data.pivot(index='ONET_SOC_CODE', columns=columns_pivot, values='DATA_VALUE')
            data = data.reset_index().set_index('ONET_SOC_CODE')
            data.columns.names = [None, None]
            df = pd.concat([df, data], axis=1)
        return df

    def load_ors_data(self):
        """Load ORS labels and scale to [0,1] via percent/100 while preserving true zeros."""
        ors_path = os.path.join(self.data_dir_ors, 'final_second_wave_2023.csv')
        ors_data = pd.read_csv(ors_path)
        ors_data.rename(columns={'SOC_2018_CODE': 'ONET_SOC_CODE', 'ESTIMATE': 'ESTIMATE_WFH_ABLE'}, inplace=True)
        ors_data['ONET_SOC_CODE'] = ors_data['ONET_SOC_CODE'] + '.00'

        # Scale ORS to [0,1] by percent/100. Keep exact zeros; cap exact ones slightly below 1.
        eps = 1e-10
        ors_data['ESTIMATE_WFH_ABLE'] = ors_data['ESTIMATE_WFH_ABLE'] / 100.0
        ors_data.loc[ors_data['ESTIMATE_WFH_ABLE'] > 1, 'ESTIMATE_WFH_ABLE'] = 1.0
        ors_data.loc[ors_data['ESTIMATE_WFH_ABLE'] < 0, 'ESTIMATE_WFH_ABLE'] = 0.0
        ors_data.loc[ors_data['ESTIMATE_WFH_ABLE'] == 1.0, 'ESTIMATE_WFH_ABLE'] = 1.0 - eps
        return ors_data

# %%
#?=========================================================================================
#? DATA PREPROCESSOR
#?=========================================================================================  
class DataPreprocessor:
    """Select metric(s) and merge ONET features with ORS labels."""
    def __init__(self, onet_data, data_loader: DataLoader):
        self.data = onet_data
        self.data_loader = data_loader

    def prepare_data(self, metric='importance', aggregation_level=None, aggregation_function=max):
        # Select metric(s) by SCALE_NAME level from MultiIndex columns (ELEMENT_ID, SCALE_NAME).
        if isinstance(metric, str):
            data = self.data.xs(metric, level=1, axis=1)
        elif isinstance(metric, (tuple, list, set)):
            pieces = []
            for m in metric:
                pieces.append(self.data.xs(m, level=1, axis=1))
            data = pd.concat(pieces, axis=1)
        else:
            raise ValueError("Metric must be a string or a collection of strings.")

        # Aggregation is intentionally disabled in this minimal version.
        if aggregation_level:
            raise NotImplementedError("Aggregation is disabled in the minimal ELEMENT_ID-only pipeline.")

        # Merge with the ORS labels.
        ors_data = self.data_loader.load_ors_data()
        data = data.merge(
            ors_data[["ONET_SOC_CODE", "ESTIMATE_WFH_ABLE"]],
            left_index=True,
            right_on='ONET_SOC_CODE',
            how='left'
        ).set_index('ONET_SOC_CODE')
        self.data = data
        return self.data

# %%
#?=========================================================================================
#? ENHANCED DATAFRAME
#?=========================================================================================
# Removed EnhancedDataFrame: focus on ELEMENT_ID-only features without extra mapping.

# %%
#?=========================================================================================
#? DATA HANDLER
#?=========================================================================================
class DataStore:
    """
    Handles storing and splitting data.

    Attributes:
        raw_data (pd.DataFrame): The complete dataset.
        labeled_data (pd.DataFrame): Rows with a non-null label.
        unlabeled_data (pd.DataFrame): Rows missing the label.
        X_train, X_test, y_train, y_test: Training and testing splits of labeled data.
    """
    def __init__(self, 
                data_list, metric = ['importance'], 
                aggregation_level=None, 
                aggregation_function=max, 
                test_size=0.2,
                random_state=42,
                data_dir_ors = DATA_DIR_ORS, 
                data_dir_onet = DATA_DIR_ONET, 
                data_dir_onet_reference = DATA_DIR_ONET_REFERENCE
                ):
        
        self.Params = {
            'data_list': data_list,
            'DataPreprocessor': {
                    'metric': metric,
                    'aggregation_level': aggregation_level,
                    'aggregation_function': aggregation_function
            },
            'test_size': test_size,
            'random_state': random_state
        }
        
        # Sub classes
        self.DataLoader = DataLoader(
            data_dir_ors = data_dir_ors,
            data_dir_onet = data_dir_onet,
            data_dir_onet_reference = data_dir_onet_reference
        )      # Data Loader instance
        self.raw_data = self.DataLoader.load_onet_data(self.Params['data_list']) # Raw data
        self.DataPreprocessor = DataPreprocessor(self.raw_data, self.DataLoader) # Data Preprocessor
        # Preprocess and merge with labels
        self.raw_data = self.DataPreprocessor.prepare_data( metric=self.Params['DataPreprocessor']['metric'],
                                                            aggregation_level=self.Params['DataPreprocessor']['aggregation_level'],
                                                            aggregation_function=self.Params['DataPreprocessor']['aggregation_function'])

        self.labeled_data = None            # Rows with a non-null label
        self.unlabeled_data = None          # Rows missing the label
        self.X_train = None                 # Training features
        self.X_test = None                  # Testing features
        self.y_train = None                 # Training labels
        self.y_test = None                  # Testing labels
        self.iz_train = None                # Training labels for Stage 1 (binary)
        self.iz_test = None                 # Testing labels for Stage 1 (binary)
        self.split_by_label()               # Split the data by label



        # Automatically split the labeled data into training and testing sets.
        self.split_train_test(test_size, random_state)
    # No EnhancedDataFrame conversion in the minimal pipeline.

    def split_by_label(self, label='ESTIMATE_WFH_ABLE'):
        """
        Splits the raw data into labeled and unlabeled data based on the presence
        of the specified label.

        Args:
            label (str): The column name to check for labels. Defaults to 'ESTIMATE_WFH_ABLE'.
        """

        self.labeled_data = self.raw_data[self.raw_data[label].notna()].copy()
        self.unlabeled_data = self.raw_data[self.raw_data[label].isna()].copy()

    def split_train_test(self, test_size=0.2, random_state=42):
        """
        Splits the labeled data into training and testing sets.

        Args:
            test_size (float): Fraction of data to use as test set.
            random_state (int): Random seed for reproducibility.
            bootstrap (bool): If True, performs bootstrap sampling on the labeled data before splitting.

        Returns:
            X_train, X_test, y_train, y_test: The training/testing splits.
        """
        X = self.labeled_data.drop(columns=["ESTIMATE_WFH_ABLE"])
        y = self.labeled_data["ESTIMATE_WFH_ABLE"]
        
        
        is_zero = (y == 0).astype(int)  # Binary target for Stage 1
        
        # Split data for classifier
        self.X_train, self.X_test, self.y_train, self.y_test, self.iz_train, self.iz_test = train_test_split(
            X, y, is_zero, test_size=test_size, random_state=random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test   

    def bootstrap_split(self, random_state=None):
        """
        Replace current train/test with a bootstrap resample.
        Updates X_train/X_test/y_train/y_test/iz_train/iz_test in-place.
        """
        # Generate a bootstrap sample from the labeled data.
        bootstrap_sample = self.labeled_data.sample(frac=1, replace=True, random_state=random_state)
        
        # Out-of-bag (OOB) indices are those that were not sampled.
        oob_indices = self.labeled_data.index.difference(bootstrap_sample.index)
        
        # If no OOB samples are found (which is unlikely for large datasets),
        # fallback to a simple random split using the provided test_size.
        if len(oob_indices) == 0:
            from sklearn.model_selection import train_test_split

            bootstrap_sample, oob_sample = train_test_split(
                self.labeled_data,
                test_size=self.Params['test_size'],
                random_state=random_state
            )
        else:
            oob_sample = self.labeled_data.loc[oob_indices]
        
        # Create training and test splits.
        X_train = bootstrap_sample.drop(columns=["ESTIMATE_WFH_ABLE"])
        y_train = bootstrap_sample["ESTIMATE_WFH_ABLE"]
        X_test = oob_sample.drop(columns=["ESTIMATE_WFH_ABLE"])
        y_test = oob_sample["ESTIMATE_WFH_ABLE"]
        
        # Create binary labels for Stage 1.
        iz_train = (y_train == 0).astype(int)
        iz_test = (y_test == 0).astype(int)
        
        # Update instance attributes.
        self.X_train, self.y_train, self.iz_train = X_train, y_train, iz_train
        self.X_test, self.y_test, self.iz_test = X_test, y_test, iz_test
        
        # return self.X_train, self.X_test, self.y_train, self.y_test

# %%
#?=============================================================================
#? PLOT MANAGER CLASS
#?=============================================================================
"""Plotting utilities removed in minimal pipeline."""

# %%
#?=============================================================================
#? MODEL PIPELINE CLASS
#?=============================================================================
class ModelPipeline:
    """
    Handles training, evaluation, prediction, and explanation of the model.
    
    Attributes:
        data (DataStore): The data container instance.
        classifier_model: The base classifier.
        regressor_model: The base regressor.
        classifier_scaler: Scaler (or transformer) for classifier features.
        regressor_scaler: Scaler (or transformer) for regressor features.
        normalize (str): Method of normalization ("logit" or other).
        zero_threshold (float): Threshold to decide when a prediction is zero.
        random_state (int): Seed for reproducibility.
        suppress_messages (bool): If True, suppress printing messages.
    """
    def __init__(
                self, 
                data: DataStore, 
                classifier_model=None, 
                regressor_model=None,
                classifier_scaler=None,
                regressor_scaler=None,
                normalize="logit",
                zero_threshold=0.8,
                random_state=42,
                suppress_messages=False
                ):
        
        self.data = data  
        self.zero_threshold = zero_threshold
        self.normalize = normalize
        self.suppress_messages = suppress_messages
        
        # Set up default models if not provided.
        if classifier_model is None:
            self.classifier_model = RandomForestClassifier(random_state=random_state)
        else:
            self.classifier_model = classifier_model

        if regressor_model is None:
            self.regressor_model = RandomForestRegressor(random_state=random_state)
        else:
            self.regressor_model = regressor_model

        # Set the scalers; if None, no scaling is applied (which suits tree-based models).
        self.classifier_scaler = classifier_scaler  # e.g., StandardScaler(), or None for trees
        self.regressor_scaler = regressor_scaler    # e.g., StandardScaler(), or None for trees

        self.calibrated_classifier = None  # Calibrated version of the classifier
        self.classifier = None             # To store the classifier (or pipeline)
        self.regressor = None              # To store the regressor (or pipeline)
        self.train_data = None             # To store training splits for later evaluation

    # No plotting in minimal pipeline.

    def train(self, include_test=False):
        """
        Trains a two-stage model:
            1. A classifier to detect zero estimates.
            2. A regressor (trained on logit-transformed non-zero data) to predict non-zero values.
        """
        # Prepare training data for classifier.
        if include_test:
            X_train = self.data.labeled_data.drop(columns=["ESTIMATE_WFH_ABLE"])
            y_train = self.data.labeled_data["ESTIMATE_WFH_ABLE"]
            iz_train = (y_train == 0).astype(int)
        else:
            X_train = self.data.X_train
            y_train = self.data.y_train
            iz_train = self.data.iz_train

        # Build classifier pipeline: include scaler if provided.
        if self.classifier_scaler is not None:
            self.classifier = Pipeline([
                ('scaler', self.classifier_scaler),
                ('classifier', self.classifier_model)
            ])
        else:
            self.classifier = self.classifier_model
        
        self.classifier.fit(X_train, iz_train)
        # Optionally, calibrate the classifier.
        # self.calibrated_classifier = CalibratedClassifierCV(self.classifier, method="isotonic", cv=3)
        # self.calibrated_classifier.fit(X_train, iz_train)
        self.calibrated_classifier = self.classifier

        # Train the regressor on non-zero data only.
        X_nonzero = self.data.X_train[self.data.iz_train != 1]
        y_nonzero = self.data.y_train[self.data.iz_train != 1]

        # Apply logit transformation to avoid predictions outside [0,1].
        if self.normalize == "logit":
            y_nonzero_norm = np.log(y_nonzero / (1 - y_nonzero))
        else:
            y_nonzero_norm = y_nonzero

        # Build regressor pipeline: include scaler if provided.
        if self.regressor_scaler is not None:
            self.regressor = Pipeline([
                ('scaler', self.regressor_scaler),
                ('regressor', self.regressor_model)
            ])
        else:
            self.regressor = self.regressor_model
        
        self.regressor.fit(X_nonzero, y_nonzero_norm)

    def _get_feature_names(self):
        return list(self.data.X_train.columns)

    def _get_feature_importances(self, model):
        # Handle plain estimator or Pipeline
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        try:
            from sklearn.pipeline import Pipeline as SkPipeline
            if isinstance(model, SkPipeline):
                for name, step in model.steps[::-1]:
                    if hasattr(step, 'feature_importances_'):
                        return step.feature_importances_
        except Exception:
            pass
        raise AttributeError('Model does not expose feature_importances_.')

    def evaluate(self, split: str = 'test', verbose: bool = True):
        """Evaluate model on a chosen split.

        split: 'train' | 'test' | 'all'
        """
        if self.suppress_messages:
            verbose = False

        if split == 'train':
            X_eval = self.data.X_train
            y_eval = self.data.y_train
            iz_eval = self.data.iz_train
        elif split == 'all':
            X_eval = self.data.labeled_data.drop(columns=["ESTIMATE_WFH_ABLE"]).copy()
            y_eval = self.data.labeled_data["ESTIMATE_WFH_ABLE"].copy()
            iz_eval = (y_eval == 0).astype(int)
        else:  # 'test'
            X_eval = self.data.X_test
            y_eval = self.data.y_test
            iz_eval = self.data.iz_test

        # Stage 1: classify zeros
        zero_probs = self.calibrated_classifier.predict_proba(X_eval)[:, 1]
        predicted_zero = (zero_probs > self.zero_threshold).astype(int)

        # Stage 2: regress on non-zero cases
        X_non_zero = X_eval.loc[predicted_zero != 1]
        if not X_non_zero.empty:
            y_nz_pred_norm = self.regressor.predict(X_non_zero)
            if self.normalize == "logit":
                y_nz_pred = 1 / (1 + np.exp(-y_nz_pred_norm))
            else:
                y_nz_pred = y_nz_pred_norm
        else:
            y_nz_pred = np.array([])

        final_pred = np.zeros(len(X_eval))
        final_pred[predicted_zero == 1] = 0
        final_pred[predicted_zero != 1] = y_nz_pred

        # Metrics
        f1 = round(f1_score(iz_eval, predicted_zero), 3)
        non_zero_mask = y_eval != 0
        mae_non_zero = round(mean_absolute_error(y_eval[non_zero_mask], final_pred[non_zero_mask]), 3)
        mae = round(mean_absolute_error(y_eval, final_pred), 3)

        def safe_corr(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        corr = round(safe_corr(y_eval, final_pred), 3)
        corr_non_zero = round(safe_corr(y_eval[non_zero_mask], final_pred[non_zero_mask]), 3)
        r2 = round(r2_score(y_eval, final_pred), 3)
        r2_non_zero = round(r2_score(y_eval[non_zero_mask], final_pred[non_zero_mask]), 3)

        if verbose:
            print("Zero-Class F1:", f1)
            print("Non-Zero MAE:", mae_non_zero, "Correlation (Non-Zero):", corr_non_zero)
            print("Overall MAE:", mae, "Correlation:", corr)
            print("R-squared:", r2, "R-squared (Non-Zero):", r2_non_zero)

        self.scores = {
            'split': split,
            'f1': f1,
            'mae_non_zero': mae_non_zero,
            'mae': mae,
            'correlation': corr,
            'correlation_non_zero': corr_non_zero,
            'r2': r2,
            'r2_non_zero': r2_non_zero,
        }

        results_df = y_eval.to_frame().assign(Predicted=final_pred)
        results_df["Actual_NonZero"] = (results_df["ESTIMATE_WFH_ABLE"] > 0).astype(int)
        results_df["Predicted_NonZero"] = (results_df["Predicted"] > 0).astype(int)
        results_df['Correct_Class'] = results_df["Actual_NonZero"] == results_df["Predicted_NonZero"]
        return results_df

    # Feature importance plotting removed in minimal pipeline.

    def predict_unlabeled(self, non_zero_cutoff=0.1, classifier_margin=0.05):
        """
        Predict unlabeled rows. If classifier is marginally confident in zero
        but regressor predicts a value above cutoff, override to non-zero.
        """
        X = self.data.unlabeled_data.drop(columns="ESTIMATE_WFH_ABLE")
        zero_probs = self.calibrated_classifier.predict_proba(X)[:, 1]
        predicted_zero = (zero_probs > self.zero_threshold).astype(int)

        # Predict for all X once.
        y_pred_norm = self.regressor.predict(X)
        if self.normalize == "logit":
            y_pred = 1 / (1 + np.exp(-y_pred_norm))
        else:
            y_pred = y_pred_norm

        # Override zero predictions if classifier is only marginally confident.
        zero_idx = np.where(predicted_zero == 1)[0]
        for idx in zero_idx:
            if (zero_probs[idx] - self.zero_threshold) < classifier_margin and y_pred[idx] > non_zero_cutoff:
                predicted_zero[idx] = 0

        final_pred = np.where(predicted_zero == 1, 0.0, y_pred)
        self.data.unlabeled_data["ESTIMATE_WFH_ABLE"] = final_pred
        return final_pred

    def export_all_results(self, out_dir: str):
        """Export combined predictions, metrics for splits, and feature importances.

        Creates files under out_dir:
          - full_occupation_predictions.csv
          - model_metrics.csv
          - classifier_feature_importance.csv
          - regressor_feature_importance.csv
        """
        os.makedirs(out_dir, exist_ok=True)

        # 1) Ensure predictions
        labeled_pred = self.evaluate(split='all', verbose=False)
        # Unlabeled predictions (if any)
        unlabeled_pred = None
        if not self.data.unlabeled_data.empty:
            preds = self.predict_unlabeled()
            unlabeled_pred = self.data.unlabeled_data[["ESTIMATE_WFH_ABLE"]].copy()
            unlabeled_pred.rename(columns={"ESTIMATE_WFH_ABLE": "Predicted"}, inplace=True)
            unlabeled_pred["ESTIMATE_WFH_ABLE"] = np.nan
            unlabeled_pred["Actual_NonZero"] = 0
            unlabeled_pred["Predicted_NonZero"] = (unlabeled_pred["Predicted"] > 0).astype(int)
            unlabeled_pred["Correct_Class"] = True  # Not applicable

        # Combine
        labeled_comb = labeled_pred[["ESTIMATE_WFH_ABLE", "Predicted", "Actual_NonZero", "Predicted_NonZero", "Correct_Class"]].copy()
        labeled_comb["is_labeled"] = 1
        if unlabeled_pred is not None:
            unlabeled_pred["is_labeled"] = 0
            combined = pd.concat([labeled_comb, unlabeled_pred], axis=0)
        else:
            combined = labeled_comb
        combined.to_csv(os.path.join(out_dir, 'full_occupation_predictions.csv'))

        # 2) Metrics for train/test/all
        metrics = []
        for split in ['train', 'test', 'all']:
            _ = self.evaluate(split=split, verbose=False)
            row = dict(self.scores)
            metrics.append(row)
        pd.DataFrame(metrics).to_csv(os.path.join(out_dir, 'model_metrics.csv'), index=False)

        # 3) Feature importances
        features = self._get_feature_names()
        # Map ELEMENT_ID -> ELEMENT_NAME
        try:
            ref_path = os.path.join(self.data.DataLoader.data_dir_onet_reference, 'CONTENT_MODEL_REFERENCE.csv')
            ref_df = pd.read_csv(ref_path)
            id_to_name = ref_df.set_index('ELEMENT_ID')['ELEMENT_NAME'].to_dict()
        except Exception:
            id_to_name = {}
        # Classifier MDI
        clf_mdi = self._get_feature_importances(self.calibrated_classifier)
        clf_imp_df = pd.DataFrame({'feature': features, 'mdi_importance': clf_mdi})
        clf_imp_df['element_id'] = clf_imp_df['feature']
        clf_imp_df['element_name'] = clf_imp_df['element_id'].map(id_to_name)
        # Classifier direction: correlation of each feature with predicted zero probability (train)
        def _safe_corr(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        zero_probs_train = self.calibrated_classifier.predict_proba(self.data.X_train)[:, 1]
        clf_corrs = []
        for col in features:
            clf_corrs.append(_safe_corr(self.data.X_train[col].values, zero_probs_train))
        clf_imp_df['corr_with_pred_zero_proba'] = clf_corrs
        clf_imp_df['direction_sign'] = np.sign(clf_imp_df['corr_with_pred_zero_proba']).astype(int)
        clf_imp_df['direction_label'] = clf_imp_df['direction_sign'].map({
            1: 'more_likely_zero (less teleworkable)',
            -1: 'less_likely_zero (more teleworkable)',
            0: 'neutral'
        })
        # Classifier permutation (F1 on train)
        clf_perm = permutation_importance(
            self.calibrated_classifier,
            self.data.X_train,
            self.data.iz_train,
            n_repeats=10,
            random_state=42,
            scoring='f1',
        )
        clf_imp_df['permutation_importance'] = clf_perm.importances_mean
        clf_imp_df['permutation_metric'] = 'f1'
        clf_imp_df.sort_values('mdi_importance', ascending=False)[
            ['feature', 'element_id', 'element_name', 'mdi_importance', 'permutation_importance', 'permutation_metric', 'corr_with_pred_zero_proba', 'direction_sign', 'direction_label']
        ].to_csv(os.path.join(out_dir, 'classifier_feature_importance.csv'), index=False)

        # Regressor MDI
        reg_mdi = self._get_feature_importances(self.regressor)
        reg_imp_df = pd.DataFrame({'feature': features, 'mdi_importance': reg_mdi})
        reg_imp_df['element_id'] = reg_imp_df['feature']
        reg_imp_df['element_name'] = reg_imp_df['element_id'].map(id_to_name)
        # Regressor permutation (R2 on non-zero, logit target if applicable)
        X_nz = self.data.X_train[self.data.iz_train != 1]
        y_nz = self.data.y_train[self.data.iz_train != 1]
        if self.normalize == 'logit':
            y_nz = np.log(y_nz / (1 - y_nz))
        reg_perm = permutation_importance(
            self.regressor,
            X_nz,
            y_nz,
            n_repeats=10,
            random_state=42,
            scoring='r2',
        )
        reg_imp_df['permutation_importance'] = reg_perm.importances_mean
        reg_imp_df['permutation_metric'] = 'r2'
        # Regressor direction: correlation of each feature with predicted WFH share among non-zero (train)
        # Use predictions on X_nz in probability space
        y_nz_pred_norm = self.regressor.predict(X_nz)
        if self.normalize == 'logit':
            y_nz_pred = 1 / (1 + np.exp(-y_nz_pred_norm))
        else:
            y_nz_pred = y_nz_pred_norm
        reg_corrs = []
        for col in features:
            # Align X_nz column
            reg_corrs.append(_safe_corr(X_nz[col].values, y_nz_pred))
        reg_imp_df['corr_with_pred_share'] = reg_corrs
        reg_imp_df['direction_sign'] = np.sign(reg_imp_df['corr_with_pred_share']).astype(int)
        reg_imp_df['direction_label'] = reg_imp_df['direction_sign'].map({
            1: 'increases_teleworkability',
            -1: 'decreases_teleworkability',
            0: 'neutral'
        })
        reg_imp_df.sort_values('mdi_importance', ascending=False)[
            ['feature', 'element_id', 'element_name', 'mdi_importance', 'permutation_importance', 'permutation_metric', 'corr_with_pred_share', 'direction_sign', 'direction_label']
        ].to_csv(os.path.join(out_dir, 'regressor_feature_importance.csv'), index=False)