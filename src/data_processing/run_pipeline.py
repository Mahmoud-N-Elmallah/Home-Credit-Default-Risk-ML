import polars as pl
import numpy as np
from pathlib import Path

def load_data(raw_paths):
    """Step 0 - Load & Organize"""
    print("Loading data lazily...")
    # Base tables fit in memory easily (~300k rows). Eager load for base preprocessing.
    train_base = pl.read_csv(raw_paths['application_train'])
    test_base = pl.read_csv(raw_paths['application_test'])
    
    # Aux tables are massive. Lazy load.
    bureau = pl.scan_csv(raw_paths['bureau'])
    bureau_balance = pl.scan_csv(raw_paths['bureau_balance'])
    prev_app = pl.scan_csv(raw_paths['previous_application'])
    pos_cash = pl.scan_csv(raw_paths['pos_cash_balance'])
    installments = pl.scan_csv(raw_paths['installments_payments'])
    cc_balance = pl.scan_csv(raw_paths['credit_card_balance'])
    
    return train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance

def preprocess_base(df: pl.DataFrame, is_train=True):
    """Step 2 - Preprocess Base Tables"""
    print(f"Preprocessing base table (is_train={is_train})...")
    # Anomaly fix
    if 'DAYS_EMPLOYED' in df.columns:
        df = df.with_columns(
            pl.when(pl.col('DAYS_EMPLOYED') == 365243).then(None).otherwise(pl.col('DAYS_EMPLOYED')).alias('DAYS_EMPLOYED')
        )
        
    # Categorical encoding
    cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
    
    # Binary
    for col in cat_cols:
        if df[col].n_unique() == 2:
            # Map binary strings to integers (e.g. 0, 1) using categorical conversion
            df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
            
    # Multi-class
    multi_cols = [col for col in cat_cols if df[col].n_unique() > 2]
    if multi_cols:
        df = df.to_dummies(multi_cols, drop_first=True)
        
    return df

def get_proportions(df: pl.LazyFrame, target_col: str, prefix: str):
    """Helper to generate expression for value counts proportions."""
    categories = df.select(target_col).drop_nulls().unique().collect().get_column(target_col).to_list()
    exprs = [
        pl.col(target_col).eq(c).mean().alias(f"{prefix}_{c}_prop")
        for c in categories
    ]
    return exprs

def agg_bureau(bureau: pl.LazyFrame, bureau_balance: pl.LazyFrame):
    """Step 3.1 - Bureau Pipeline"""
    print("Aggregating bureau...")
    # Stage A
    status_exprs = get_proportions(bureau_balance, 'STATUS', 'STATUS')
    bb_agg = bureau_balance.group_by('SK_ID_BUREAU').agg(
        [
            pl.count('MONTHS_BALANCE').alias('MONTHS_BALANCE_count'),
            pl.col('MONTHS_BALANCE').min().alias('MONTHS_BALANCE_min'),
            pl.col('MONTHS_BALANCE').max().alias('MONTHS_BALANCE_max'),
        ] + status_exprs
    )
    
    # Stage B
    bureau = bureau.join(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Stage C
    bb_cols = ['MONTHS_BALANCE_count', 'MONTHS_BALANCE_min', 'MONTHS_BALANCE_max'] + [e.meta.output_name() for e in status_exprs]
    bb_mean_exprs = [pl.col(c).mean().alias(f"{c}_mean") for c in bb_cols]
    
    b_agg = bureau.group_by('SK_ID_CURR').agg(
        [
            pl.col('DAYS_CREDIT').min().alias('DAYS_CREDIT_min'),
            pl.col('DAYS_CREDIT').max().alias('DAYS_CREDIT_max'),
            pl.col('DAYS_CREDIT').mean().alias('DAYS_CREDIT_mean'),
            pl.col('CREDIT_DAY_OVERDUE').max().alias('CREDIT_DAY_OVERDUE_max'),
            pl.col('CREDIT_DAY_OVERDUE').mean().alias('CREDIT_DAY_OVERDUE_mean'),
            pl.col('AMT_CREDIT_SUM').sum().alias('AMT_CREDIT_SUM_sum'),
            pl.col('AMT_CREDIT_SUM').mean().alias('AMT_CREDIT_SUM_mean'),
            pl.col('AMT_CREDIT_SUM_DEBT').sum().alias('AMT_CREDIT_SUM_DEBT_sum'),
            pl.col('AMT_CREDIT_SUM_DEBT').mean().alias('AMT_CREDIT_SUM_DEBT_mean'),
            pl.col('AMT_CREDIT_SUM_OVERDUE').sum().alias('AMT_CREDIT_SUM_OVERDUE_sum'),
            pl.col('CREDIT_ACTIVE').eq('Active').mean().alias('CREDIT_ACTIVE_prop_active')
        ] + bb_mean_exprs
    )
    
    rename_dict = {col: f"bureau_{col}" for col in b_agg.collect_schema().names() if col != 'SK_ID_CURR'}
    return b_agg.rename(rename_dict)

def agg_prev_app(prev_app: pl.LazyFrame):
    """Step 3.2 - Previous Applications"""
    print("Aggregating previous apps...")
    days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols:
        prev_app = prev_app.with_columns(
            pl.when(pl.col(col) == 365243).then(None).otherwise(pl.col(col)).alias(col)
        )
        
    status_exprs = get_proportions(prev_app, 'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_STATUS')
    type_exprs = get_proportions(prev_app, 'NAME_PRODUCT_TYPE', 'NAME_PRODUCT_TYPE')
    
    agg = prev_app.group_by('SK_ID_CURR').agg(
        [
            pl.count('SK_ID_PREV').alias('SK_ID_PREV_count'),
            pl.col('AMT_ANNUITY').min().alias('AMT_ANNUITY_min'),
            pl.col('AMT_ANNUITY').max().alias('AMT_ANNUITY_max'),
            pl.col('AMT_ANNUITY').mean().alias('AMT_ANNUITY_mean'),
            pl.col('AMT_APPLICATION').min().alias('AMT_APPLICATION_min'),
            pl.col('AMT_APPLICATION').max().alias('AMT_APPLICATION_max'),
            pl.col('AMT_APPLICATION').mean().alias('AMT_APPLICATION_mean'),
            pl.col('AMT_CREDIT').min().alias('AMT_CREDIT_min'),
            pl.col('AMT_CREDIT').max().alias('AMT_CREDIT_max'),
            pl.col('AMT_CREDIT').mean().alias('AMT_CREDIT_mean'),
            pl.col('AMT_DOWN_PAYMENT').mean().alias('AMT_DOWN_PAYMENT_mean'),
            pl.col('DAYS_DECISION').min().alias('DAYS_DECISION_min'),
            pl.col('DAYS_DECISION').max().alias('DAYS_DECISION_max'),
            pl.col('DAYS_DECISION').mean().alias('DAYS_DECISION_mean')
        ] + status_exprs + type_exprs
    )
    
    rename_dict = {col: f"prev_{col}" for col in agg.collect_schema().names() if col != 'SK_ID_CURR'}
    return agg.rename(rename_dict)

def agg_pos_cash(pos_cash: pl.LazyFrame):
    """Step 3.3 - POS CASH Balance"""
    print("Aggregating POS cash...")
    status_exprs = get_proportions(pos_cash, 'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_STATUS')
    agg = pos_cash.group_by('SK_ID_CURR').agg(
        [
            pl.col('SK_ID_PREV').n_unique().alias('SK_ID_PREV_nunique'),
            pl.col('MONTHS_BALANCE').min().alias('MONTHS_BALANCE_min'),
            pl.col('MONTHS_BALANCE').max().alias('MONTHS_BALANCE_max'),
            pl.count('MONTHS_BALANCE').alias('MONTHS_BALANCE_count'),
            pl.col('CNT_INSTALMENT').mean().alias('CNT_INSTALMENT_mean'),
            pl.col('CNT_INSTALMENT_FUTURE').mean().alias('CNT_INSTALMENT_FUTURE_mean'),
            pl.col('SK_DPD').max().alias('SK_DPD_max'),
            pl.col('SK_DPD').mean().alias('SK_DPD_mean'),
            pl.col('SK_DPD_DEF').max().alias('SK_DPD_DEF_max'),
            pl.col('SK_DPD_DEF').mean().alias('SK_DPD_DEF_mean'),
        ] + status_exprs
    )
    rename_dict = {col: f"pos_{col}" for col in agg.collect_schema().names() if col != 'SK_ID_CURR'}
    return agg.rename(rename_dict)

def agg_installments(installments: pl.LazyFrame):
    """Step 3.4 - Installments Payments"""
    print("Aggregating installments...")
    installments = installments.with_columns([
        (pl.col('AMT_PAYMENT') / (pl.col('AMT_INSTALMENT') + 1e-8)).alias('PAYMENT_PERC'),
        (pl.col('AMT_INSTALMENT') - pl.col('AMT_PAYMENT')).alias('PAYMENT_DIFF'),
        (pl.col('DAYS_ENTRY_PAYMENT') - pl.col('DAYS_INSTALMENT')).clip(lower_bound=0).alias('DPD'),
        (pl.col('DAYS_INSTALMENT') - pl.col('DAYS_ENTRY_PAYMENT')).clip(lower_bound=0).alias('DBD')
    ])
    
    agg = installments.group_by('SK_ID_CURR').agg([
        pl.col('NUM_INSTALMENT_VERSION').n_unique().alias('NUM_INSTALMENT_VERSION_nunique'),
        pl.col('DPD').max().alias('DPD_max'),
        pl.col('DPD').mean().alias('DPD_mean'),
        pl.col('DPD').sum().alias('DPD_sum'),
        pl.col('DBD').max().alias('DBD_max'),
        pl.col('DBD').mean().alias('DBD_mean'),
        pl.col('DBD').sum().alias('DBD_sum'),
        pl.col('PAYMENT_PERC').mean().alias('PAYMENT_PERC_mean'),
        pl.col('PAYMENT_PERC').var().alias('PAYMENT_PERC_var'),
        pl.col('PAYMENT_DIFF').mean().alias('PAYMENT_DIFF_mean'),
        pl.col('PAYMENT_DIFF').var().alias('PAYMENT_DIFF_var'),
        pl.col('AMT_INSTALMENT').mean().alias('AMT_INSTALMENT_mean'),
        pl.col('AMT_INSTALMENT').sum().alias('AMT_INSTALMENT_sum'),
        pl.col('AMT_PAYMENT').mean().alias('AMT_PAYMENT_mean'),
        pl.col('AMT_PAYMENT').sum().alias('AMT_PAYMENT_sum'),
        pl.col('DAYS_ENTRY_PAYMENT').max().alias('DAYS_ENTRY_PAYMENT_max')
    ])
    rename_dict = {col: f"inst_{col}" for col in agg.collect_schema().names() if col != 'SK_ID_CURR'}
    return agg.rename(rename_dict)

def agg_cc_balance(cc_balance: pl.LazyFrame):
    """Step 3.5 - Credit Card Balance"""
    print("Aggregating CC balance...")
    cc_balance = cc_balance.with_columns(
        (pl.col('AMT_BALANCE') / (pl.col('AMT_CREDIT_LIMIT_ACTUAL') + 1e-8)).clip(0, 1).alias('UTILIZATION')
    )
    
    agg = cc_balance.group_by('SK_ID_CURR').agg([
        pl.col('SK_ID_PREV').n_unique().alias('SK_ID_PREV_nunique'),
        pl.col('MONTHS_BALANCE').min().alias('MONTHS_BALANCE_min'),
        pl.col('MONTHS_BALANCE').max().alias('MONTHS_BALANCE_max'),
        pl.count('MONTHS_BALANCE').alias('MONTHS_BALANCE_count'),
        pl.col('AMT_BALANCE').max().alias('AMT_BALANCE_max'),
        pl.col('AMT_BALANCE').mean().alias('AMT_BALANCE_mean'),
        pl.col('AMT_CREDIT_LIMIT_ACTUAL').max().alias('AMT_CREDIT_LIMIT_ACTUAL_max'),
        pl.col('AMT_CREDIT_LIMIT_ACTUAL').mean().alias('AMT_CREDIT_LIMIT_ACTUAL_mean'),
        pl.col('AMT_DRAWINGS_CURRENT').sum().alias('AMT_DRAWINGS_CURRENT_sum'),
        pl.col('AMT_DRAWINGS_CURRENT').mean().alias('AMT_DRAWINGS_CURRENT_mean'),
        pl.col('AMT_PAYMENT_TOTAL_CURRENT').sum().alias('AMT_PAYMENT_TOTAL_CURRENT_sum'),
        pl.col('AMT_PAYMENT_TOTAL_CURRENT').mean().alias('AMT_PAYMENT_TOTAL_CURRENT_mean'),
        pl.col('CNT_DRAWINGS_CURRENT').sum().alias('CNT_DRAWINGS_CURRENT_sum'),
        pl.col('CNT_DRAWINGS_CURRENT').mean().alias('CNT_DRAWINGS_CURRENT_mean'),
        pl.col('SK_DPD').max().alias('SK_DPD_max'),
        pl.col('SK_DPD').mean().alias('SK_DPD_mean'),
        pl.col('SK_DPD_DEF').max().alias('SK_DPD_DEF_max'),
        pl.col('SK_DPD_DEF').mean().alias('SK_DPD_DEF_mean'),
        pl.col('UTILIZATION').mean().alias('UTILIZATION_mean'),
        pl.col('UTILIZATION').max().alias('UTILIZATION_max')
    ])
    rename_dict = {col: f"cc_{col}" for col in agg.collect_schema().names() if col != 'SK_ID_CURR'}
    return agg.rename(rename_dict)

def merge_all(base: pl.DataFrame, aggs: dict, name="", high_null_threshold=0.6):
    """Step 4 - Sequential Merge"""
    print(f"Merging tables for {name}...")
    df = base.lazy()
    for agg_name, agg_df in aggs.items():
        df = df.join(agg_df, on='SK_ID_CURR', how='left')
        
    df = df.collect() # Trigger lazy execution
    print(f"  Shape after all merges: {df.shape}")
    
    # Check high nulls
    null_rates = [df.get_column(col).null_count() / df.height for col in df.columns]
    high_nulls = [col for col, rate in zip(df.columns, null_rates) if rate > high_null_threshold and col not in base.columns]
    if high_nulls:
        print(f"  Warning: {len(high_nulls)} aggregated columns with high null rate (>{high_null_threshold}). Sample: {high_nulls[:3]}")
            
    return df

def impute_missing(train: pl.DataFrame, test: pl.DataFrame, aux_prefixes):
    """Step 5 - Missing Value Imputation"""
    print("Imputing missing values...")
    
    # 1. Aux columns -> 0
    aux_cols = [c for c in train.columns if any(c.startswith(p) for p in aux_prefixes)]
    train = train.with_columns([pl.col(c).fill_null(0) for c in aux_cols])
    
    test_aux_cols = [c for c in test.columns if c in aux_cols]
    if test_aux_cols:
        test = test.with_columns([pl.col(c).fill_null(0) for c in test_aux_cols])
            
    # 2. Add indicators for base cols + median impute
    base_cols = [c for c in train.columns if c not in aux_cols and c not in ['TARGET', 'SK_ID_CURR'] and train.get_column(c).dtype in pl.NUMERIC_DTYPES]
    
    # Calculate medians on train
    medians = train.select([pl.col(c).median() for c in base_cols])
    
    train_exprs = []
    test_exprs = []
    
    for col in base_cols:
        med_val = medians.get_column(col)[0]
        # Skip string types or things that can't be null evaluated
        if med_val is None:
            continue
            
        train_has_null = train.get_column(col).null_count() > 0
        test_has_null = test.get_column(col).null_count() > 0 if col in test.columns else False
        
        if train_has_null:
            train_exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_is_missing"))
            train_exprs.append(pl.col(col).fill_null(med_val))
            
            if col in test.columns:
                test_exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_is_missing"))
                test_exprs.append(pl.col(col).fill_null(med_val))
        elif test_has_null:
            # fill in test even if train had no nulls
            test_exprs.append(pl.col(col).fill_null(med_val))
            
    if train_exprs:
        train = train.with_columns(train_exprs)
    if test_exprs:
        test = test.with_columns(test_exprs)
        
    return train, test

def add_global_features(df: pl.DataFrame):
    """Step 6 - Global Feature Engineering"""
    print("Adding global features...")
    
    exprs = [
        (pl.col('AMT_CREDIT') / (pl.col('AMT_INCOME_TOTAL') + 1e-8)).alias('CREDIT_INCOME_RATIO'),
        (pl.col('AMT_ANNUITY') / (pl.col('AMT_INCOME_TOTAL') + 1e-8)).alias('ANNUITY_INCOME_RATIO'),
        (pl.col('AMT_ANNUITY') / (pl.col('AMT_CREDIT') + 1e-8)).alias('CREDIT_TERM')
    ]
    
    if 'AMT_GOODS_PRICE' in df.columns:
        exprs.append((pl.col('AMT_GOODS_PRICE') / (pl.col('AMT_CREDIT') + 1e-8)).alias('GOODS_CREDIT_RATIO'))
        
    if 'bureau_AMT_CREDIT_SUM_DEBT_sum' in df.columns:
        exprs.append((pl.col('bureau_AMT_CREDIT_SUM_DEBT_sum') / (pl.col('AMT_INCOME_TOTAL') + 1e-8)).alias('BUREAU_DEBT_RATIO'))
        exprs.append((pl.col('bureau_AMT_CREDIT_SUM_DEBT_sum') / (pl.col('bureau_AMT_CREDIT_SUM_sum') + 1e-8)).alias('BUREAU_DEBT_CREDIT_RATIO'))
        
    if 'inst_AMT_PAYMENT_sum' in df.columns:
        exprs.append((pl.col('inst_AMT_PAYMENT_sum') / (pl.col('inst_AMT_INSTALMENT_sum') + 1e-8)).alias('INSTALMENT_PAYMENT_RATIO'))
        
    if 'prev_NAME_CONTRACT_STATUS_Approved_prop' in df.columns:
        exprs.append((pl.col('prev_NAME_CONTRACT_STATUS_Approved_prop') / (pl.col('prev_SK_ID_PREV_count') + 1e-8)).alias('PREV_APPROVAL_RATE'))
        
    df = df.with_columns(exprs)
    
    # Fix inf/nan
    num_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES]
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).fill_null(0).alias(c)
        for c in num_cols
    ])
    
    return df

def feature_cleanup(train: pl.DataFrame, test: pl.DataFrame, corr_threshold=0.95, var_threshold=0.995):
    """Step 7 - Feature Cleanup"""
    print("Cleaning up features...")
    # Drop correlation
    num_cols = [c for c in train.columns if train[c].dtype in pl.NUMERIC_DTYPES and c not in ['TARGET', 'SK_ID_CURR']]
    
    print("  Computing correlation matrix...")
    train_pd = train.select(num_cols).to_pandas()
    corr_matrix = train_pd.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    print(f"  Dropping {len(to_drop_corr)} correlated cols")
    
    train = train.drop(to_drop_corr, strict=False)
    
    # Drop near zero var
    to_drop_var = []
    for col in train.columns:
        if col not in ['TARGET', 'SK_ID_CURR']:
            # get the highest frequency proportion
            counts = train.get_column(col).value_counts()
            max_prop = counts.get_column("count").max() / train.height
            if max_prop > var_threshold:
                to_drop_var.append(col)
                
    print(f"  Dropping {len(to_drop_var)} low var cols")
    train = train.drop(to_drop_var, strict=False)
    
    # Align
    test_cols = [c for c in train.columns if c != 'TARGET']
    
    missing_in_test = [c for c in test_cols if c not in test.columns]
    if missing_in_test:
        test = test.with_columns([pl.lit(0, dtype=pl.UInt8).alias(c) for c in missing_in_test])
        
    test = test.select(test_cols)
    
    return train, test

def validate(train: pl.DataFrame, test: pl.DataFrame):
    """Step 8 - Validation"""
    print("Validating...")
    errors = []
    
    if 'TARGET' in test.columns: errors.append("TARGET in test")
    if 'TARGET' not in train.columns: errors.append("TARGET not in train")
    
    diff = list(set(train.columns) - set(test.columns))
    if diff != ['TARGET']: errors.append(f"Column diff mismatch: {diff}")
    
    # null check
    if sum(test.null_count().row(0)) > 0: errors.append("Test has nulls")
    if sum(train.null_count().row(0)) > 0: errors.append("Train has nulls")
    
    if train.get_column('SK_ID_CURR').n_unique() != train.height: errors.append("Train ID not unique")
    if test.get_column('SK_ID_CURR').n_unique() != test.height: errors.append("Test ID not unique")
    
    if errors:
        print("VALIDATION FAILED:")
        for e in errors: print(f" - {e}")
        return False
        
    print("Validation passed.")
    print(f"Final Train Shape: {train.shape}")
    print(f"Final Test Shape: {test.shape}")
    return True

def run_pipeline(config):
    try:
        train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance = load_data(config['data']['raw'])
    except Exception as e:
        print(f"Data missing or error: {e}")
        return
    
    # Preprocess
    train_base = preprocess_base(train_base, is_train=True)
    test_base = preprocess_base(test_base, is_train=False)
    
    # Agg
    b_agg = agg_bureau(bureau, bureau_balance)
    p_agg = agg_prev_app(prev_app)
    pos_agg = agg_pos_cash(pos_cash)
    inst_agg = agg_installments(installments)
    cc_agg = agg_cc_balance(cc_balance)
    
    aggs = {
        'bureau': b_agg,
        'prev': p_agg,
        'pos': pos_agg,
        'inst': inst_agg,
        'cc': cc_agg
    }
    
    # Merge
    high_null_threshold = config.get('pipeline', {}).get('high_null_threshold', 0.6)
    train_full = merge_all(train_base, aggs, "Train", high_null_threshold)
    test_full = merge_all(test_base, aggs, "Test", high_null_threshold)
    
    # Impute
    prefixes = ['bureau_', 'prev_', 'pos_', 'inst_', 'cc_']
    train_full, test_full = impute_missing(train_full, test_full, prefixes)
    
    # Global feats
    train_full = add_global_features(train_full)
    test_full = add_global_features(test_full)
    
    # Cleanup
    corr_threshold = config.get('pipeline', {}).get('correlation_threshold', 0.95)
    var_threshold = config.get('pipeline', {}).get('variance_threshold', 0.995)
    train_full, test_full = feature_cleanup(train_full, test_full, corr_threshold, var_threshold)
    
    # Validate & Export
    if validate(train_full, test_full):
        print("Exporting...")
        final_paths = config['data']['final']
        
        Path(final_paths['train']).parent.mkdir(parents=True, exist_ok=True)
        Path(final_paths['test']).parent.mkdir(parents=True, exist_ok=True)
        
        train_full.write_csv(final_paths['train'])
        test_full.write_csv(final_paths['test'])
        print(f"Done. Saved to {final_paths['train']} and {final_paths['test']}.")
