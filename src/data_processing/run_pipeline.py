from pathlib import Path

from src.data_processing.aggregations import (
    agg_bureau,
    agg_cc_balance,
    agg_installments,
    agg_pos_cash,
    agg_prev_app,
)
from src.data_processing.features import (
    add_global_features,
    feature_cleanup,
    impute_missing,
    merge_all,
    preprocess_base,
    validate,
)
from src.data_processing.io import latest_submission_path, load_data, write_feature_manifest


def run_pipeline(config):
    train_base, test_base, bureau, bureau_balance, prev_app, pos_cash, installments, cc_balance = load_data(
        config["data"]["raw"],
        config,
    )
    train_base, test_base = preprocess_base(train_base, test_base, config)
    b_agg = agg_bureau(bureau, bureau_balance, config)
    p_agg = agg_prev_app(prev_app, config)
    pos_agg = agg_pos_cash(pos_cash, config)
    inst_agg = agg_installments(installments, config)
    cc_agg = agg_cc_balance(cc_balance, config)
    aggs = {"bureau": b_agg, "prev": p_agg, "pos": pos_agg, "inst": inst_agg, "cc": cc_agg}

    train_full = merge_all(train_base, aggs, config, "Train")
    test_full = merge_all(test_base, aggs, config, "Test")
    train_full, test_full = impute_missing(train_full, test_full, config)
    train_full = add_global_features(train_full, config)
    test_full = add_global_features(test_full, config)
    train_full, test_full, cleanup_info = feature_cleanup(train_full, test_full, config)
    validate(train_full, test_full, config)
    write_feature_manifest(train_full, test_full, config, cleanup_info)

    final_paths = config["data"]["final"]
    Path(final_paths["train"]).parent.mkdir(parents=True, exist_ok=True)
    train_full.write_csv(final_paths["train"])
    test_full.write_csv(final_paths["test"])
    submission_path = latest_submission_path(config)
    if config["pipeline"]["warn_on_stale_submission"] and submission_path is not None and submission_path.exists():
        print(f"Warning: existing submission may be stale after processing. Regenerate it with --train: {submission_path}")
    print("Data processing done.")
