import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import optuna

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Sklearn & Imblearn
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

def clean_column_names(df):
    """LightGBM doesn't support JSON characters in column names."""
    df.columns = [re.sub(r'[^\w]', '_', c) for c in df.columns]
    return df

def get_scaler(scaler_name):
    if scaler_name == 'standard': return StandardScaler()
    elif scaler_name == 'robust': return RobustScaler()
    elif scaler_name == 'minmax': return MinMaxScaler()
    return None

def get_imbalance_sampler(strategy, config):
    seed = config['globals']['random_state']
    if strategy == 'smote': return SMOTE(random_state=seed)
    elif strategy == 'borderline_smote': return BorderlineSMOTE(random_state=seed)
    elif strategy == 'adasyn': return ADASYN(random_state=seed)
    elif strategy == 'random_undersample': return RandomUnderSampler(random_state=seed)
    return None

def get_model(name, params, class_weight_strategy, config, is_trial=True):
    params = params.copy() if params else {}
    seed = config['globals']['random_state']
    ratio = config['training'].get('imbalance_ratio', 1)
    
    if class_weight_strategy == 'class_weight':
        if name == 'lightgbm': params['is_unbalance'] = True
        elif name == 'xgboost': params['scale_pos_weight'] = ratio
        elif name == 'catboost': params['auto_class_weights'] = 'Balanced'

    if name == 'lightgbm':
        return LGBMClassifier(random_state=seed, **params)
    elif name == 'xgboost':
        return XGBClassifier(random_state=seed, **params)
    elif name == 'catboost':
        v_key = 'catboost_trial' if is_trial else 'catboost_final'
        v_val = config['training']['verbosity'].get(v_key, 0)
        return CatBoostClassifier(random_state=seed, verbose=v_val, **params)
    else:
        raise ValueError(f"Unknown model: {name}")

def calculate_metric(y_true, y_pred, metric_name, threshold):
    if metric_name == 'roc_auc': return roc_auc_score(y_true, y_pred)
    elif metric_name == 'average_precision': return average_precision_score(y_true, y_pred)
    elif metric_name == 'f1':
        y_pred_bin = (y_pred > threshold).astype(int)
        return f1_score(y_true, y_pred_bin)
    raise ValueError(f"Unknown metric: {metric_name}")

def save_evaluation_report(y_true, y_pred_prob, model_name, models_dir, threshold):
    y_pred_bin = (y_pred_prob > threshold).astype(int)
    report_str = f"Model: {model_name}\n"
    report_str += "="*40 + "\n"
    report_str += f"Classification Threshold: {threshold}\n"
    report_str += f"ROC AUC Score: {roc_auc_score(y_true, y_pred_prob):.4f}\n"
    report_str += f"Average Precision (PR AUC): {average_precision_score(y_true, y_pred_prob):.4f}\n"
    report_str += f"F1 Score: {f1_score(y_true, y_pred_bin):.4f}\n\n"
    report_str += "Classification Report:\n"
    report_str += classification_report(y_true, y_pred_bin)

    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / f"{model_name}_evaluation_report.txt", "w") as f:
        f.write(report_str)

    cm = confusion_matrix(y_true, y_pred_bin)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name} (Thresh: {threshold})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(models_dir / f"{model_name}_confusion_matrix.png")
    plt.close()

def suggest_params(trial, search_space):
    params = {}
    for param_name, space in search_space.items():
        if space['type'] == 'int':
            params[param_name] = trial.suggest_int(param_name, space['low'], space['high'], log=space.get('log', False))
        elif space['type'] == 'float':
            params[param_name] = trial.suggest_float(param_name, space['low'], space['high'], log=space.get('log', False))
        elif space['type'] == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, space['choices'])
    return params

def run_training(config):
    print("Initializing Optimized Training Pipeline...")
    t_config = config['training']
    seed = config['globals']['random_state']
    models_dir = Path(t_config['artifact_paths']['models_dir'])
    
    # 1. Load Data
    train_path = Path(config['data']['final']['train'])
    if not train_path.exists(): raise FileNotFoundError(f"Training data not found at {train_path}.")
    df = pd.read_csv(train_path)
    y = df[t_config['target_col']]
    X = df.drop(columns=[t_config['target_col'], t_config['id_col']])
    X = clean_column_names(X)

    # 2. Pre-Scaling
    scaler_name = t_config['preprocessing']['scaler']
    scaler = get_scaler(scaler_name)
    if scaler:
        print(f"  [STEP] Pre-Scaling data ({scaler_name.upper()})...")
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        joblib.dump(scaler, models_dir / f"{scaler_name}_scaler.pkl")

    # 3. Pre-Feature Selection
    fs_config = t_config['preprocessing'].get('feature_selection', {'method': 'none'})
    if fs_config['method'] == 'lgbm':
        max_f = fs_config.get('max_features', 150)
        fs_params = fs_config.get('selector_params', {})
        print(f"  [STEP] Pre-Selecting top {max_f} features using LightGBM...")
        lgb_sel = LGBMClassifier(random_state=seed, **fs_params)
        selector = SelectFromModel(lgb_sel, max_features=max_f, threshold=-np.inf)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()]
        X = X[selected_cols]
        joblib.dump(selector, models_dir / "feature_selector.pkl")
    else:
        selector = None

    # 4. Handle Mode
    if t_config['models']['type'] == 'single':
        run_single_optuna(X, y, config, models_dir, selector)
    elif t_config['models']['type'] == 'ensemble':
        run_ensemble_optimized(X, y, config, models_dir, selector)

def run_single_optuna(X, y, config, models_dir, selector):
    t_config = config['training']
    seed = config['globals']['random_state']
    subsample_rate = t_config.get('optuna_subsample_rate', 1.0)
    
    est_config = t_config['models']['estimators'][0]
    name = est_config['name']
    search_space = est_config.get('search_space', {})
    
    # Stratified Subsampling for speed
    if subsample_rate < 1.0:
        print(f"  [STEP] Subsampling {subsample_rate*100}% of data for Optuna speed...")
        X_search, _, y_search, _ = train_test_split(X, y, train_size=subsample_rate, stratify=y, random_state=seed)
    else:
        X_search, y_search = X, y

    def objective(trial):
        trial_params = suggest_params(trial, search_space)
        model_params = {k.replace('model__', ''): v for k, v in trial_params.items()}
        params = est_config.get('params', {}).copy()
        params.update(model_params)
        
        model = get_model(name, params, t_config['preprocessing']['imbalance_strategy'], config, is_trial=True)
        
        # Handle resampler if not class_weight
        strategy = t_config['preprocessing']['imbalance_strategy']
        sampler = get_imbalance_sampler(strategy, config)
        
        cv = StratifiedKFold(n_splits=t_config['cv_splits'], shuffle=t_config['cv_shuffle'], random_state=seed)
        
        scores = []
        for tr_idx, val_idx in cv.split(X_search, y_search):
            X_tr, y_tr = X_search.iloc[tr_idx], y_search.iloc[tr_idx]
            X_va, y_va = X_search.iloc[val_idx], y_search.iloc[val_idx]
            
            if sampler: X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
            
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(calculate_metric(y_va, preds, t_config['optimization_metric'], t_config['classification_threshold']))
            
        return np.mean(scores)

    study = optuna.create_study(direction=t_config['optuna_direction'], sampler=optuna.samplers.TPESampler(seed=seed))
    print(f"Starting Optuna search for {name}...")
    study.optimize(objective, n_trials=t_config['optuna_n_trials'], show_progress_bar=True)
    
    best_params_clean = {k.replace('model__', ''): v for k, v in study.best_params.items()}
    with open(models_dir / f"{name}_best_params.yaml", 'w') as f:
        yaml.dump(best_params_clean, f)
        
    print(f"Retraining best model on 100% data...")
    final_params = est_config.get('params', {}).copy()
    final_params.update(best_params_clean)
    best_model = get_model(name, final_params, t_config['preprocessing']['imbalance_strategy'], config, is_trial=False)
    
    X_final, y_final = X, y
    sampler = get_imbalance_sampler(t_config['preprocessing']['imbalance_strategy'], config)
    if sampler: X_final, y_final = sampler.fit_resample(X_final, y_final)
    
    best_model.fit(X_final, y_final)
    joblib.dump(best_model, models_dir / f"{name}_optuna_best_model.pkl")
    
    y_pred_prob = best_model.predict_proba(X)[:, 1]
    save_evaluation_report(y, y_pred_prob, f"{name}_optuna", models_dir, t_config['classification_threshold'])
    
    # Global predict call for submission
    predict_test_and_submit(best_model, config, is_ensemble=False)

def run_ensemble_optimized(X, y, config, models_dir, selector):
    t_config = config['training']
    seed = config['globals']['random_state']
    threshold = t_config['classification_threshold']
    
    cv = StratifiedKFold(n_splits=t_config['cv_splits'], shuffle=t_config['cv_shuffle'], random_state=seed)
    estimators = t_config['models']['estimators']
    oof_preds = np.zeros(len(X))
    
    sampler = get_imbalance_sampler(t_config['preprocessing']['imbalance_strategy'], config)
    
    print("\nStarting CV Loop...")
    for tr_idx, va_idx in tqdm(cv.split(X, y), total=t_config['cv_splits'], desc="Ensemble Folds"):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        
        if sampler: X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
            
        fold_blend_preds = np.zeros(len(X_va))
        total_weight = sum([est['weight'] for est in estimators])
        
        for est_conf in estimators:
            model = get_model(est_conf['name'], est_conf.get('params', {}), t_config['preprocessing']['imbalance_strategy'], config, is_trial=False)
            model.fit(X_tr, y_tr)
            fold_blend_preds += model.predict_proba(X_va)[:, 1] * (est_conf['weight'] / total_weight)
            
        oof_preds[va_idx] = fold_blend_preds
        
    save_evaluation_report(y, oof_preds, "ensemble_cv_oof", models_dir, threshold)
    
    print("\nRetraining all models on 100% data...")
    X_full, y_full = X, y
    if sampler: X_full, y_full = sampler.fit_resample(X_full, y_full)
    
    final_models = []
    for est_conf in estimators:
        model = get_model(est_conf['name'], est_conf.get('params', {}), t_config['preprocessing']['imbalance_strategy'], config, is_trial=False)
        model.fit(X_full, y_full)
        joblib.dump(model, models_dir / f"{est_conf['name']}_ensemble_model.pkl")
        final_models.append((model, est_conf['weight']))
        
    predict_test_and_submit(final_models, config, is_ensemble=True)

def predict_test_and_submit(model_obj, config, is_ensemble=False):
    print("\nGenerating Predictions...")
    test_path = Path(config['data']['final']['test'])
    if not test_path.exists(): return
    df_test = pd.read_csv(test_path)
    ids = df_test[config['training']['id_col']]
    X_test = clean_column_names(df_test.drop(columns=[config['training']['id_col']]))
    
    # Apply artifacts
    models_dir = Path(config['training']['artifact_paths']['models_dir'])
    scaler_name = config['training']['preprocessing']['scaler']
    scaler_path = models_dir / f"{scaler_name}_scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
    selector_path = models_dir / "feature_selector.pkl"
    if selector_path.exists():
        selector = joblib.load(selector_path)
        X_test = X_test[X_test.columns[selector.get_support()]]
    
    if not is_ensemble:
        preds = model_obj.predict_proba(X_test)[:, 1]
    else:
        preds = np.zeros(len(X_test))
        total_weight = sum([w for _, w in model_obj])
        for model, weight in model_obj:
            preds += model.predict_proba(X_test)[:, 1] * (weight / total_weight)

    sub = pd.DataFrame({config['training']['id_col']: ids, config['training']['target_col']: preds})
    sub.to_csv(config['training']['artifact_paths']['submission'], index=False)
    print(f"Submission saved to {config['training']['artifact_paths']['submission']}")
