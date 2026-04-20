import polars as pl
import numpy as np
from pathlib import Path

def load_data(raw_paths):
    """Step 0 - Load & Organize"""
    print("Loading data lazily...")
    # Increase schema inference length to handle large datasets with initial nulls
    train_base = pl.read_csv(raw_paths['application_train'], infer_schema_length=10000)
    test_base = pl.read_csv(raw_paths['application_test'], infer_schema_length=10000)
    
    bureau = pl.scan_csv(raw_paths['bureau'], infer_schema_length=10000)
    bureau_balance = pl.scan_csv(raw_paths['bureau_balance'], infer_schema_length=10000)
    prev_app = pl.scan_csv(raw_paths['previous_application'], infer_schema_length=10000)
    pos_cash = pl.scan_csv(raw_paths['pos_cash_balance'], infer_schema_length=10000)
    installments = pl.scan_csv(raw_paths['installments_payments'], infer_schema_length=10000)
    cc_balance = pl.scan_csv(raw_paths['credit_card_balance'], infer_schema_length=10000)
    
    return train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance

def get_proportions(df: pl.LazyFrame, target_col: str, prefix: str):
    """Helper to generate expression for value counts proportions."""
    categories = df.select(target_col).drop_nulls().unique().collect().get_column(target_col).to_list()
    exprs = [
        pl.col(target_col).eq(c).mean().alias(f"{prefix}_{c}_prop")
        for c in categories
    ]
    return exprs

def preprocess_base(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 2 - Preprocess Base Tables (Enhanced)"""
    print("Preprocessing base tables with advanced feature engineering...")
    fe_config = config['pipeline']['feature_engineering']
    anomaly_val = config['pipeline']['anomaly_fix']['days_employed']
    
    def transform_base(df: pl.DataFrame, is_train=True):
        # 1. Anomaly fix
        if 'DAYS_EMPLOYED' in df.columns:
            df = df.with_columns(
                pl.when(pl.col('DAYS_EMPLOYED') == anomaly_val).then(None).otherwise(pl.col('DAYS_EMPLOYED')).alias('DAYS_EMPLOYED')
            )
        
        # 2. Domain Ratios
        if all(c in df.columns for c in ['DAYS_EMPLOYED', 'DAYS_BIRTH']):
            df = df.with_columns((pl.col('DAYS_EMPLOYED') / pl.col('DAYS_BIRTH')).alias('DAYS_EMPLOYED_PERC'))
        if all(c in df.columns for c in ['AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS']):
            df = df.with_columns((pl.col('AMT_INCOME_TOTAL') / pl.col('CNT_FAM_MEMBERS')).alias('INCOME_PER_PERSON'))
            
        # 3. Document Sum
        doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
        if doc_cols:
            df = df.with_columns(pl.sum_horizontal(doc_cols).alias('DOCUMENT_COUNT'))
            
        # 4. Categorical encoding
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String and col not in fe_config['frequency_encoding_cols']]
        
        # Binary
        for col in cat_cols:
            if df[col].n_unique() == 2:
                df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
                
        # Multi-class
        multi_cols = [col for col in cat_cols if df[col].n_unique() > 2]
        if multi_cols:
            df = df.to_dummies(multi_cols, drop_first=True)
            
        return df

    # Frequency Encoding (Anti-Leakage)
    for col in fe_config['frequency_encoding_cols']:
        if col in train.columns:
            freqs = train.get_column(col).value_counts()
            train = train.join(freqs, on=col, how='left').rename({'count': f'{col}_FREQ'})
            test = test.join(freqs, on=col, how='left').rename({'count': f'{col}_FREQ'}).with_columns(pl.col(f'{col}_FREQ').fill_null(1))
            # Drop original
            train = train.drop(col)
            test = test.drop(col)

    train = transform_base(train, is_train=True)
    test = transform_base(test, is_train=False)
    
    return train, test

def agg_bureau(bureau: pl.LazyFrame, bureau_balance: pl.LazyFrame, config):
    """Step 3.1 - Bureau Pipeline (Enhanced)"""
    print("Aggregating bureau with recency...")
    id_curr = config['training']['id_col']
    fe_config = config['pipeline']['feature_engineering']
    
    # Cast known numeric for safety
    bureau = bureau.with_columns([
        pl.col('DAYS_CREDIT').cast(pl.Float64),
        pl.col('AMT_CREDIT_SUM').cast(pl.Float64),
        pl.col('AMT_CREDIT_SUM_DEBT').cast(pl.Float64)
    ])
    bureau_balance = bureau_balance.with_columns(pl.col('MONTHS_BALANCE').cast(pl.Float64))

    # Stage A: BB Agg
    status_exprs = get_proportions(bureau_balance, 'STATUS', 'STATUS')
    bb_agg = bureau_balance.group_by('SK_ID_BUREAU').agg(
        [
            pl.count('MONTHS_BALANCE').alias('MONTHS_BALANCE_count'),
            pl.col('MONTHS_BALANCE').min().alias('MONTHS_BALANCE_min'),
            pl.col('MONTHS_BALANCE').max().alias('MONTHS_BALANCE_max'),
        ] + status_exprs
    )
    
    # Recency for BB
    for m in fe_config['recency_months']:
        bb_recent = bureau_balance.filter(pl.col('MONTHS_BALANCE') >= -m)
        status_exprs_recent = get_proportions(bb_recent, 'STATUS', f'STATUS_{m}M')
        bb_agg_recent = bb_recent.group_by('SK_ID_BUREAU').agg(status_exprs_recent)
        bb_agg = bb_agg.join(bb_agg_recent, on='SK_ID_BUREAU', how='left')
    
    # Stage B
    bureau = bureau.join(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Stage C: Bureau Agg
    bb_agg_cols = [c for c in bb_agg.collect_schema().names() if c != 'SK_ID_BUREAU']
    bb_mean_exprs = [pl.col(c).mean().alias(f"{c}_mean") for c in bb_agg_cols]
    
    active_val = fe_config['categories']['bureau_active']
    
    b_agg = bureau.group_by(id_curr).agg(
        [
            pl.col('DAYS_CREDIT').min().alias('DAYS_CREDIT_min'),
            pl.col('DAYS_CREDIT').max().alias('DAYS_CREDIT_max'),
            pl.col('DAYS_CREDIT').mean().alias('DAYS_CREDIT_mean'),
            pl.col('AMT_CREDIT_SUM').sum().alias('AMT_CREDIT_SUM_sum'),
            pl.col('AMT_CREDIT_SUM_DEBT').sum().alias('AMT_CREDIT_SUM_DEBT_sum'),
            pl.col('CREDIT_ACTIVE').eq(active_val).sum().alias('ACTIVE_LOANS_COUNT'),
            pl.col('CREDIT_ACTIVE').eq(active_val).mean().alias('CREDIT_ACTIVE_prop_active')
        ] + bb_mean_exprs
    )
    
    rename_dict = {col: f"bureau_{col}" for col in b_agg.collect_schema().names() if col != id_curr}
    return b_agg.rename(rename_dict)

def agg_prev_app(prev_app: pl.LazyFrame, config):
    """Step 3.2 - Previous Applications (Enhanced)"""
    print("Aggregating previous apps with gap and recency...")
    id_curr = config['training']['id_col']
    fe_config = config['pipeline']['feature_engineering']
    anomaly_val = config['pipeline']['anomaly_fix']['days_employed']
    
    # Cast
    prev_app = prev_app.with_columns([
        pl.col('AMT_APPLICATION').cast(pl.Float64),
        pl.col('AMT_CREDIT').cast(pl.Float64),
        pl.col('AMT_ANNUITY').cast(pl.Float64),
        pl.col('DAYS_DECISION').cast(pl.Float64)
    ])

    days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in days_cols:
        prev_app = prev_app.with_columns(
            pl.when(pl.col(col) == anomaly_val).then(None).otherwise(pl.col(col)).alias(col)
        )
    
    # Gap analysis
    prev_app = prev_app.with_columns((pl.col('AMT_APPLICATION') - pl.col('AMT_CREDIT')).alias('APP_CREDIT_GAP'))
    
    status_exprs = get_proportions(prev_app, 'NAME_CONTRACT_STATUS', 'NAME_CONTRACT_STATUS')
    
    agg = prev_app.group_by(id_curr).agg(
        [
            pl.count('SK_ID_PREV').alias('SK_ID_PREV_count'),
            pl.col('AMT_ANNUITY').mean().alias('AMT_ANNUITY_mean'),
            pl.col('AMT_CREDIT').sum().alias('AMT_CREDIT_sum'),
            pl.col('APP_CREDIT_GAP').sum().alias('APP_CREDIT_GAP_sum'),
            pl.col('APP_CREDIT_GAP').mean().alias('APP_CREDIT_GAP_mean'),
            pl.col('DAYS_DECISION').min().alias('DAYS_DECISION_min'),
            pl.col('DAYS_DECISION').mean().alias('DAYS_DECISION_mean')
        ] + status_exprs
    )
    
    # Recency for Prev App
    for d in fe_config['recency_days_prev']:
        recent = prev_app.filter(pl.col('DAYS_DECISION') >= -d)
        recent_agg = recent.group_by(id_curr).agg([
            pl.count('SK_ID_PREV').alias(f'SK_ID_PREV_count_{d}D'),
            pl.col('APP_CREDIT_GAP').mean().alias(f'APP_CREDIT_GAP_mean_{d}D')
        ])
        agg = agg.join(recent_agg, on=id_curr, how='left')
    
    rename_dict = {col: f"prev_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)

def agg_pos_cash(pos_cash: pl.LazyFrame, config):
    """Step 3.3 - POS CASH Balance (Enhanced)"""
    print("Aggregating POS cash with recency...")
    id_curr = config['training']['id_col']
    fe_config = config['pipeline']['feature_engineering']
    
    pos_cash = pos_cash.with_columns([
        pl.col('MONTHS_BALANCE').cast(pl.Float64),
        pl.col('SK_DPD').cast(pl.Float64)
    ])

    agg = pos_cash.group_by(id_curr).agg([
        pl.col('SK_ID_PREV').n_unique().alias('SK_ID_PREV_nunique'),
        pl.col('SK_DPD').max().alias('SK_DPD_max'),
        pl.col('SK_DPD').mean().alias('SK_DPD_mean')
    ])
    
    for m in fe_config['recency_months']:
        recent = pos_cash.filter(pl.col('MONTHS_BALANCE') >= -m)
        recent_agg = recent.group_by(id_curr).agg([
            pl.col('SK_DPD').max().alias(f'SK_DPD_max_{m}M'),
            pl.col('SK_DPD').mean().alias(f'SK_DPD_mean_{m}M')
        ])
        agg = agg.join(recent_agg, on=id_curr, how='left')
        
    rename_dict = {col: f"pos_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)

def agg_installments(installments: pl.LazyFrame, config):
    """Step 3.4 - Installments Payments"""
    print("Aggregating installments...")
    id_curr = config['training']['id_col']
    eps = config['globals']['division_epsilon']
    
    # Cast
    installments = installments.with_columns([
        pl.col('AMT_PAYMENT').cast(pl.Float64),
        pl.col('AMT_INSTALMENT').cast(pl.Float64),
        pl.col('DAYS_ENTRY_PAYMENT').cast(pl.Float64),
        pl.col('DAYS_INSTALMENT').cast(pl.Float64)
    ])

    installments = installments.with_columns([
        (pl.col('AMT_PAYMENT') / (pl.col('AMT_INSTALMENT') + eps)).alias('PAYMENT_PERC'),
        (pl.col('AMT_INSTALMENT') - pl.col('AMT_PAYMENT')).alias('PAYMENT_DIFF'),
        (pl.col('DAYS_ENTRY_PAYMENT') - pl.col('DAYS_INSTALMENT')).clip(lower_bound=0).alias('DPD'),
        (pl.col('DAYS_INSTALMENT') - pl.col('DAYS_ENTRY_PAYMENT')).clip(lower_bound=0).alias('DBD')
    ])
    
    agg = installments.group_by(id_curr).agg([
        pl.col('NUM_INSTALMENT_VERSION').n_unique().alias('NUM_INSTALMENT_VERSION_nunique'),
        pl.col('DPD').max().alias('DPD_max'),
        pl.col('DPD').mean().alias('DPD_mean'),
        pl.col('PAYMENT_PERC').mean().alias('PAYMENT_PERC_mean'),
        pl.col('AMT_INSTALMENT').sum().alias('AMT_INSTALMENT_sum'),
        pl.col('AMT_PAYMENT').sum().alias('AMT_PAYMENT_sum')
    ])
    rename_dict = {col: f"inst_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)

def agg_cc_balance(cc_balance: pl.LazyFrame, config):
    """Step 3.5 - Credit Card Balance (Enhanced)"""
    print("Aggregating CC balance with recency...")
    id_curr = config['training']['id_col']
    eps = config['globals']['division_epsilon']
    fe_config = config['pipeline']['feature_engineering']
    
    # Cast
    cc_balance = cc_balance.with_columns([
        pl.col('AMT_BALANCE').cast(pl.Float64),
        pl.col('AMT_CREDIT_LIMIT_ACTUAL').cast(pl.Float64),
        pl.col('AMT_DRAWINGS_CURRENT').cast(pl.Float64),
        pl.col('MONTHS_BALANCE').cast(pl.Float64),
        pl.col('SK_DPD').cast(pl.Float64)
    ])

    cc_balance = cc_balance.with_columns(
        (pl.col('AMT_BALANCE') / (pl.col('AMT_CREDIT_LIMIT_ACTUAL') + eps)).clip(0, 1).alias('UTILIZATION')
    )
    
    agg = cc_balance.group_by(id_curr).agg([
        pl.col('AMT_BALANCE').mean().alias('AMT_BALANCE_mean'),
        pl.col('AMT_DRAWINGS_CURRENT').sum().alias('AMT_DRAWINGS_CURRENT_sum'),
        pl.col('SK_DPD').max().alias('SK_DPD_max')
    ])
    
    for m in fe_config['recency_months']:
        recent = cc_balance.filter(pl.col('MONTHS_BALANCE') >= -m)
        recent_agg = recent.group_by(id_curr).agg([
            pl.col('AMT_DRAWINGS_CURRENT').sum().alias(f'AMT_DRAWINGS_CURRENT_sum_{m}M'),
            pl.col('SK_DPD').max().alias(f'SK_DPD_max_{m}M')
        ])
        agg = agg.join(recent_agg, on=id_curr, how='left')
        
    rename_dict = {col: f"cc_{col}" for col in agg.collect_schema().names() if col != id_curr}
    return agg.rename(rename_dict)

def merge_all(base: pl.DataFrame, aggs: dict, config, name=""):
    """Step 4 - Sequential Merge"""
    print(f"Merging tables for {name}...")
    id_curr = config['training']['id_col']
    df = base.lazy()
    for agg_name, agg_df in aggs.items():
        df = df.join(agg_df, on=id_curr, how='left')
        
    df = df.collect() # Trigger lazy execution
    print(f"  Shape after all merges: {df.shape}")
    
    # Check high nulls
    high_null_threshold = config['pipeline']['high_null_threshold']
    null_rates = [df.get_column(col).null_count() / df.height for col in df.columns]
    high_nulls = [col for col, rate in zip(df.columns, null_rates) if rate > high_null_threshold and col not in base.columns]
    if high_nulls:
        print(f"  Warning: {len(high_nulls)} aggregated columns with high null rate (>{high_null_threshold}). Sample: {high_nulls[:3]}")
            
    return df

def impute_missing(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 5 - Missing Value Imputation"""
    print("Imputing missing values...")
    aux_prefixes = config['pipeline']['aux_table_prefixes']
    target_col = config['training']['target_col']
    id_col = config['training']['id_col']
    
    # 1. Aux columns -> 0
    aux_cols = [c for c in train.columns if any(c.startswith(p) for p in aux_prefixes)]
    train = train.with_columns([pl.col(c).fill_null(0) for c in aux_cols])
    
    test_aux_cols = [c for c in test.columns if c in aux_cols]
    if test_aux_cols:
        test = test.with_columns([pl.col(c).fill_null(0) for c in test_aux_cols])
            
    # 2. Add indicators for base cols + median impute
    base_cols = [c for c in train.columns if c not in aux_cols and c not in [target_col, id_col] and train.get_column(c).dtype in pl.NUMERIC_DTYPES]
    
    # Calculate medians on train
    medians = train.select([pl.col(c).median() for c in base_cols])
    
    train_exprs = []
    test_exprs = []
    
    for col in base_cols:
        med_val = medians.get_column(col)[0]
        if med_val is None: continue
            
        train_has_null = train.get_column(col).null_count() > 0
        test_has_null = test.get_column(col).null_count() > 0 if col in test.columns else False
        
        if train_has_null:
            train_exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_is_missing"))
            train_exprs.append(pl.col(col).fill_null(med_val))
            
            if col in test.columns:
                test_exprs.append(pl.col(col).is_null().cast(pl.Int32).alias(f"{col}_is_missing"))
                test_exprs.append(pl.col(col).fill_null(med_val))
        elif test_has_null:
            test_exprs.append(pl.col(col).fill_null(med_val))
            
    if train_exprs: train = train.with_columns(train_exprs)
    if test_exprs: test = test.with_columns(test_exprs)
        
    return train, test

def add_global_features(df: pl.DataFrame, config):
    """Step 6 - Global Feature Engineering (Enhanced)"""
    print("Adding global features including EXT interactions...")
    eps = config['globals']['division_epsilon']
    fe_config = config['pipeline']['feature_engineering']
    
    exprs = [
        (pl.col('AMT_CREDIT') / (pl.col('AMT_INCOME_TOTAL') + eps)).alias('CREDIT_INCOME_RATIO'),
        (pl.col('AMT_ANNUITY') / (pl.col('AMT_INCOME_TOTAL') + eps)).alias('ANNUITY_INCOME_RATIO'),
        (pl.col('AMT_ANNUITY') / (pl.col('AMT_CREDIT') + eps)).alias('CREDIT_TERM')
    ]
    
    # EXT Interactions
    ext_cols = fe_config['ext_sources']
    if all(c in df.columns for c in ext_cols):
        exprs.append(pl.mean_horizontal(ext_cols).alias('EXT_SOURCES_MEAN'))
        exprs.append((pl.col(ext_cols[0]) * pl.col(ext_cols[1]) * pl.col(ext_cols[2])).alias('EXT_SOURCES_PROD'))
        # Manual std implementation as mean_horizontal is available but horizontal std might be different in versions
        mean = pl.mean_horizontal(ext_cols)
        exprs.append(pl.sqrt(pl.mean_horizontal([(pl.col(c) - mean)**2 for c in ext_cols])).alias('EXT_SOURCES_STD'))

    # Enquiry Total
    enq_cols = fe_config['enquiry_cols']
    if all(c in df.columns for c in enq_cols):
        exprs.append(pl.sum_horizontal(enq_cols).alias('AMT_REQ_CREDIT_BUREAU_TOTAL'))

    if 'AMT_GOODS_PRICE' in df.columns:
        exprs.append((pl.col('AMT_GOODS_PRICE') / (pl.col('AMT_CREDIT') + eps)).alias('GOODS_CREDIT_RATIO'))
        
    if 'bureau_AMT_CREDIT_SUM_DEBT_sum' in df.columns:
        exprs.append((pl.col('bureau_AMT_CREDIT_SUM_DEBT_sum') / (pl.col('AMT_INCOME_TOTAL') + eps)).alias('BUREAU_DEBT_RATIO'))
        exprs.append((pl.col('bureau_AMT_CREDIT_SUM_DEBT_sum') / (pl.col('bureau_AMT_CREDIT_SUM_sum') + eps)).alias('BUREAU_DEBT_CREDIT_RATIO'))
        
    if 'inst_AMT_PAYMENT_sum' in df.columns:
        exprs.append((pl.col('inst_AMT_PAYMENT_sum') / (pl.col('inst_AMT_INSTALMENT_sum') + eps)).alias('INSTALMENT_PAYMENT_RATIO'))
        
    if 'prev_NAME_CONTRACT_STATUS_Approved_prop' in df.columns:
        exprs.append((pl.col('prev_NAME_CONTRACT_STATUS_Approved_prop') / (pl.col('prev_SK_ID_PREV_count') + eps)).alias('PREV_APPROVAL_RATE'))
        
    df = df.with_columns(exprs)
    
    # Fix inf/nan
    num_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES]
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).fill_null(0).alias(c)
        for c in num_cols
    ])
    
    return df

def feature_cleanup(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 7 - Feature Cleanup"""
    print("Cleaning up features...")
    corr_threshold = config['pipeline']['correlation_threshold']
    var_threshold = config['pipeline']['variance_threshold']
    target_col = config['training']['target_col']
    id_col = config['training']['id_col']
    
    num_cols = [c for c in train.columns if train[c].dtype in pl.NUMERIC_DTYPES and c not in [target_col, id_col]]
    train_pd = train.select(num_cols).to_pandas()
    corr_matrix = train_pd.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    print(f"  Dropping {len(to_drop_corr)} correlated cols")
    train = train.drop(to_drop_corr, strict=False)
    
    to_drop_var = []
    for col in train.columns:
        if col not in [target_col, id_col]:
            counts = train.get_column(col).value_counts()
            max_prop = counts.get_column("count").max() / train.height
            if max_prop > var_threshold:
                to_drop_var.append(col)
                
    print(f"  Dropping {len(to_drop_var)} low var cols")
    train = train.drop(to_drop_var, strict=False)
    
    test_cols = [c for c in train.columns if c != target_col]
    missing_in_test = [c for c in test_cols if c not in test.columns]
    if missing_in_test:
        test = test.with_columns([pl.lit(0, dtype=pl.UInt8).alias(c) for c in missing_in_test])
    test = test.select(test_cols)
    
    return train, test

def validate(train: pl.DataFrame, test: pl.DataFrame, config):
    """Step 8 - Validation"""
    print("Validating...")
    errors = []
    target_col = config['training']['target_col']
    id_col = config['training']['id_col']
    
    if target_col in test.columns: errors.append(f"{target_col} in test")
    if target_col not in train.columns: errors.append(f"{target_col} not in train")
    if sum(test.null_count().row(0)) > 0: errors.append("Test has nulls")
    if sum(train.null_count().row(0)) > 0: errors.append("Train has nulls")
    if train.get_column(id_col).n_unique() != train.height: errors.append("Train ID not unique")
    
    if errors:
        print("VALIDATION FAILED:")
        for e in errors: print(f" - {e}")
        return False
        
    print(f"Validation passed. Train: {train.shape}, Test: {test.shape}")
    return True

def run_pipeline(config):
    try:
        train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance = load_data(config['data']['raw'])
    except Exception as e:
        print(f"Data missing or error: {e}")
        return
    
    train_base, test_base = preprocess_base(train_base, test_base, config)
    
    b_agg = agg_bureau(bureau, bureau_balance, config)
    p_agg = agg_prev_app(prev_app, config)
    pos_agg = agg_pos_cash(pos_cash, config)
    inst_agg = agg_installments(installments, config)
    cc_agg = agg_cc_balance(cc_balance, config)
    
    aggs = {'bureau': b_agg, 'prev': p_agg, 'pos': pos_agg, 'inst': inst_agg, 'cc': cc_agg}
    
    train_full = merge_all(train_base, aggs, config, "Train")
    test_full = merge_all(test_base, aggs, config, "Test")
    
    train_full, test_full = impute_missing(train_full, test_full, config)
    train_full = add_global_features(train_full, config)
    test_full = add_global_features(test_full, config)
    
    train_full, test_full = feature_cleanup(train_full, test_full, config)
    
    if validate(train_full, test_full, config):
        final_paths = config['data']['final']
        Path(final_paths['train']).parent.mkdir(parents=True, exist_ok=True)
        train_full.write_csv(final_paths['train'])
        test_full.write_csv(final_paths['test'])
        print("Done.")
