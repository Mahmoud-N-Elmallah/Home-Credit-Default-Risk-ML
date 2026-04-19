import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Sklearn & Imblearn
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def clean_column_names(df):
    """LightGBM doesn't support JSON characters in column names."""
    df.columns = [re.sub(r'[^\w]', '_', c) for c in df.columns]
    return df

def get_feature_selector(method, max_features):
    if method == 'lgbm':
        print(f"  [INIT] Selecting top {max_features} features using LightGBM (CPU)...")
        # Use a quick LGBM model to evaluate feature importance
        lgb_sel = LGBMClassifier(n_estimators=50, random_state=42, verbose=-1, n_jobs=-1)
        # threshold=-np.inf forces it to keep exactly max_features
        return SelectFromModel(lgb_sel, max_features=max_features, threshold=-np.inf)
    return 'passthrough'

def save_evaluation_report(y_true, y_pred_prob, model_name, models_dir):
    """Save metrics, classification report and confusion matrix artifact."""
    y_pred_bin = (y_pred_prob > 0.5).astype(int)

    # 1. Text Report
    report_str = f"Model: {model_name}\n"
    report_str += "="*40 + "\n"
    report_str += f"ROC AUC Score: {roc_auc_score(y_true, y_pred_prob):.4f}\n"
    report_str += f"Average Precision (PR AUC): {average_precision_score(y_true, y_pred_prob):.4f}\n"
    report_str += f"F1 Score: {f1_score(y_true, y_pred_bin):.4f}\n\n"
    report_str += "Classification Report:\n"
    report_str += classification_report(y_true, y_pred_bin)

    report_path = models_dir / f"{model_name}_evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    plot_path = models_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"  [SAVED] Evaluation report and confusion matrix to {models_dir}")

def get_scaler(scaler_name):
    print(f"  [INIT] Scaler: {scaler_name.upper()}")
    if scaler_name == 'standard':
        return StandardScaler()
    elif scaler_name == 'robust':
        return RobustScaler()
    elif scaler_name == 'minmax':
        return MinMaxScaler()
    return 'passthrough'

def get_imbalance_sampler(strategy):
    print(f"  [INIT] Imbalance Strategy: {strategy.upper()}")
    if strategy == 'smote':
        return SMOTE(random_state=42)
    elif strategy == 'borderline_smote':
        return BorderlineSMOTE(random_state=42)
    elif strategy == 'adasyn':
        return ADASYN(random_state=42)
    return 'passthrough'

def get_model(name, params, class_weight_strategy):
    params = params.copy() if params else {}
    
    if class_weight_strategy == 'class_weight':
        if name in ['lightgbm', 'xgboost']:
            params['scale_pos_weight'] = 11.3 # Approx imbalance ratio for Home Credit (92% to 8%)
        elif name == 'catboost':
            params['auto_class_weights'] = 'Balanced'

    if name == 'lightgbm':
        device = params.get('device_type', 'cpu')
        print(f"  [INIT] Model: LightGBM (Device: {device.upper()})")
        return LGBMClassifier(random_state=42, **params)
    elif name == 'xgboost':
        device = params.get('device', 'cpu')
        print(f"  [INIT] Model: XGBoost (Device: {device.upper()})")
        return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)
    elif name == 'catboost':
        device = params.get('task_type', 'CPU')
        print(f"  [INIT] Model: CatBoost (Device: {device.upper()})")
        return CatBoostClassifier(random_state=42, verbose=50, **params)
    else:
        raise ValueError(f"Unknown model: {name}")

def calculate_metric(y_true, y_pred, metric_name):
    if metric_name == 'roc_auc':
        return roc_auc_score(y_true, y_pred)
    elif metric_name == 'average_precision':
        return average_precision_score(y_true, y_pred)
    elif metric_name == 'f1':
        y_pred_bin = (y_pred > 0.5).astype(int)
        return f1_score(y_true, y_pred_bin)
    raise ValueError(f"Unknown metric: {metric_name}")

def run_single_model(X, y, config, models_dir):
    print("Running Single Model Mode (GridSearchCV)")
    X = clean_column_names(X)
    
    est_config = config['models']['estimators'][0]
    name = est_config['name']
    
    # Extract params and setup grid
    base_params = est_config.get('params', {})
    param_grid = est_config.get('param_grid', {})
    
    scaler = get_scaler(config['preprocessing']['scaler'])
    sampler = get_imbalance_sampler(config['preprocessing']['imbalance_strategy'])
    
    fs_config = config['preprocessing'].get('feature_selection', {'method': 'none'})
    selector = get_feature_selector(fs_config['method'], fs_config.get('max_features', 150))
    
    model = get_model(name, base_params, config['preprocessing']['imbalance_strategy'])
    
    # Imblearn Pipeline guarantees no leakage during CV
    pipeline = ImbPipeline([
        ('scaler', scaler),
        ('sampler', sampler),
        ('selector', selector),
        ('model', model)
    ])
    
    cv = StratifiedKFold(n_splits=config['cv_splits'], shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=config['optimization_metric'],
        cv=cv,
        verbose=3,
        n_jobs=-1 if name != 'catboost' else 1 # Catboost handles its own threads better
    )
    
    print(f"Starting GridSearchCV for {name}...")
    import warnings
    warnings.filterwarnings('ignore', message='.*valid feature names.*')
    grid.fit(X, y)
    
    print(f"\nBest {config['optimization_metric']} Score: {grid.best_score_:.4f}")
    print(f"Best Parameters:\n{grid.best_params_}")
    
    # Save artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save params
    with open(models_dir / f"{name}_best_params.yaml", 'w') as f:
        yaml.dump(grid.best_params_, f)
        
    # Save model pipeline
    best_estimator = grid.best_estimator_
    joblib.dump(best_estimator, models_dir / f"{name}_grid_best_model.pkl")
    print(f"Saved best model and params to {models_dir}")
    
    # Predict and evaluate on full training data just for a final report artifact
    y_pred_prob = best_estimator.predict_proba(X)[:, 1]
    save_evaluation_report(y, y_pred_prob, f"{name}_grid", models_dir)
    
    return best_estimator

def run_ensemble(X, y, config, models_dir):
    print("Running Ensemble Mode (Weighted Blending)")
    X = clean_column_names(X)
    
    cv = StratifiedKFold(n_splits=config['cv_splits'], shuffle=True, random_state=42)
    
    estimators = config['models']['estimators']
    oof_preds = np.zeros(len(X))
    metric_name = config['optimization_metric']
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold + 1} ---")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Scale
        scaler = get_scaler(config['preprocessing']['scaler'])
        if scaler != 'passthrough':
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_va = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns, index=X_va.index)
            
        # 2. Resample (Only on train!)
        sampler = get_imbalance_sampler(config['preprocessing']['imbalance_strategy'])
        if sampler != 'passthrough':
            X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
            
        # 3. Feature Selection
        fs_config = config['preprocessing'].get('feature_selection', {'method': 'none'})
        selector = get_feature_selector(fs_config['method'], fs_config.get('max_features', 150))
        if selector != 'passthrough':
            selector.fit(X_tr, y_tr)
            X_tr = pd.DataFrame(selector.transform(X_tr), columns=X_tr.columns[selector.get_support()], index=X_tr.index)
            X_va = pd.DataFrame(selector.transform(X_va), columns=X_va.columns[selector.get_support()], index=X_va.index)
            
        # 4. Train models and blend
        fold_blend_preds = np.zeros(len(X_va))
        total_weight = sum([est['weight'] for est in estimators])
        
        for est_conf in estimators:
            name = est_conf['name']
            weight = est_conf['weight'] / total_weight
            
            print(f"  Training {name} (Weight: {weight:.2f})...")
            model = get_model(name, est_conf.get('params', {}), config['preprocessing']['imbalance_strategy'])
            
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            fold_blend_preds += preds * weight
            
        oof_preds[val_idx] = fold_blend_preds
        score = calculate_metric(y_va, fold_blend_preds, metric_name)
        print(f"  Fold {fold + 1} Blended {metric_name}: {score:.4f}")
        fold_scores.append(score)
        
    print(f"\nOverall CV {metric_name}: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    
    # Save Out-Of-Fold Evaluation Report
    models_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_report(y, oof_preds, "ensemble_cv_oof", models_dir)
    
    # --- Retrain on FULL dataset for final artifacts ---
    print("\nRetraining on FULL train dataset for final export...")
    scaler = get_scaler(config['preprocessing']['scaler'])
    if scaler != 'passthrough':
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        joblib.dump(scaler, models_dir / f"{config['preprocessing']['scaler']}_scaler.pkl")
        
    sampler = get_imbalance_sampler(config['preprocessing']['imbalance_strategy'])
    if sampler != 'passthrough':
        X, y = sampler.fit_resample(X, y)
        
    fs_config = config['preprocessing'].get('feature_selection', {'method': 'none'})
    selector = get_feature_selector(fs_config['method'], fs_config.get('max_features', 150))
    if selector != 'passthrough':
        selector.fit(X, y)
        X = pd.DataFrame(selector.transform(X), columns=X.columns[selector.get_support()], index=X.index)
        joblib.dump(selector, models_dir / "feature_selector.pkl")
        
    final_models = []
    for est_conf in estimators:
        name = est_conf['name']
        model = get_model(name, est_conf.get('params', {}), config['preprocessing']['imbalance_strategy'])
        model.fit(X, y)
        joblib.dump(model, models_dir / f"{name}_ensemble_model.pkl")
        final_models.append((model, est_conf['weight']))
        print(f"  Saved {name}_ensemble_model.pkl")
        
    return final_models, scaler, selector

def predict_test_and_submit(final_model_obj, config):
    print("\nGenerating Predictions on Test Set...")
    test_path = Path(config['data']['final']['test'])
    if not test_path.exists():
        print(f"Test file not found at {test_path}")
        return
        
    df_test = pd.read_csv(test_path)
    id_col = config['training']['id_col']
    ids = df_test[id_col]
    X_test = df_test.drop(columns=[id_col])
    X_test = clean_column_names(X_test)
    
    if config['training']['models']['type'] == 'single':
        # final_model_obj is an Imblearn Pipeline
        preds = final_model_obj.predict_proba(X_test)[:, 1]
        
    elif config['training']['models']['type'] == 'ensemble':
        # final_model_obj is (list_of_tuples(model, weight), scaler, selector)
        models_weights, scaler, selector = final_model_obj
        if scaler != 'passthrough':
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            
        if selector != 'passthrough':
            X_test = pd.DataFrame(selector.transform(X_test), columns=X_test.columns[selector.get_support()], index=X_test.index)
            
        preds = np.zeros(len(X_test))
        total_weight = sum([w for _, w in models_weights])
        for model, weight in models_weights:
            norm_w = weight / total_weight
            preds += model.predict_proba(X_test)[:, 1] * norm_w

    # Save
    sub = pd.DataFrame({
        id_col: ids,
        config['training']['target_col']: preds
    })
    
    sub_path = Path(config['data']['final']['train']).parent / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

def run_training(config):
    print("Initializing Training Pipeline...")
    t_config = config['training']
    
    train_path = Path(config['data']['final']['train'])
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}. Run processing first.")
        
    df_train = pd.read_csv(train_path)
    
    target_col = t_config['target_col']
    id_col = t_config['id_col']
    
    y = df_train[target_col]
    X = df_train.drop(columns=[target_col, id_col])
    
    models_dir = Path("Models")
    
    if t_config['models']['type'] == 'single':
        final_model = run_single_model(X, y, t_config, models_dir)
        predict_test_and_submit(final_model, config)
        
    elif t_config['models']['type'] == 'ensemble':
        final_model = run_ensemble(X, y, t_config, models_dir)
        predict_test_and_submit(final_model, config)
    else:
        print(f"Unknown model type: {t_config['models']['type']}")
