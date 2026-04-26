"""Fixed Home Credit source schema and feature definitions."""

SCHEMA_OVERRIDES = {
    "SK_ID_CURR": "Int64",
    "SK_ID_BUREAU": "Int64",
    "SK_ID_PREV": "Int64",
    "AMT_INCOME_TOTAL": "Float64",
    "AMT_CREDIT": "Float64",
    "AMT_ANNUITY": "Float64",
    "AMT_GOODS_PRICE": "Float64",
    "AMT_PAYMENT": "Float64",
    "AMT_INSTALMENT": "Float64",
    "DAYS_ENTRY_PAYMENT": "Float64",
    "DAYS_INSTALMENT": "Float64",
    "AMT_BALANCE": "Float64",
    "AMT_CREDIT_LIMIT_ACTUAL": "Float64",
    "AMT_DRAWINGS_CURRENT": "Float64",
    "AMT_APPLICATION": "Float64",
    "DAYS_DECISION": "Float64",
    "DAYS_EMPLOYED": "Float64",
    "DAYS_BIRTH": "Float64",
    "DAYS_REGISTRATION": "Float64",
    "DAYS_ID_PUBLISH": "Float64",
    "DAYS_LAST_PHONE_CHANGE": "Float64",
    "MONTHS_BALANCE": "Float64",
    "DAYS_CREDIT": "Float64",
    "DAYS_CREDIT_ENDDATE": "Float64",
    "DAYS_ENDDATE_FACT": "Float64",
    "AMT_CREDIT_SUM": "Float64",
    "AMT_CREDIT_SUM_DEBT": "Float64",
    "AMT_CREDIT_SUM_LIMIT": "Float64",
    "AMT_CREDIT_SUM_OVERDUE": "Float64",
    "CNT_INSTALMENT": "Float64",
    "CNT_INSTALMENT_FUTURE": "Float64",
    "CNT_PAYMENT": "Float64",
    "AMT_DOWN_PAYMENT": "Float64",
    "RATE_DOWN_PAYMENT": "Float64",
    "AMT_TOTAL_RECEIVABLE": "Float64",
    "AMT_RECEIVABLE_PRINCIPAL": "Float64",
    "AMT_RECIVABLE": "Float64",
    "CNT_DRAWINGS_CURRENT": "Float64",
    "SK_DPD": "Float64",
    "SK_DPD_DEF": "Float64",
}

DAY_UNIT = 365.25
DAYS_EMPLOYED_COL = "DAYS_EMPLOYED"
DOCUMENT_FLAG_PREFIX = "FLAG_DOCUMENT_"
LAST_N_RECORDS = [1, 3, 5, 10]
RECENCY_MONTHS = [3, 6, 12, 24]
RECENCY_DAYS = [60, 90, 180, 365, 730, 1000]
BUREAU_RECENCY_DAYS = [365, 730, 1095]
FREQUENCY_ENCODING_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]
EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
EXT_SOURCE_INTERACTION_BASES = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]
MISSING_INDICATOR_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_LAST_PHONE_CHANGE",
]
ENQUIRY_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

APPLICATION_SOURCE_COLS = {
    "days_birth": "DAYS_BIRTH",
    "amt_goods_price": "AMT_GOODS_PRICE",
    "amt_credit": "AMT_CREDIT",
}

OUTPUT_FEATURES = {
    "document_count": "DOCUMENT_COUNT",
    "ext_sources_mean": "EXT_SOURCES_MEAN",
    "ext_sources_prod": "EXT_SOURCES_PROD",
    "ext_sources_std": "EXT_SOURCES_STD",
    "ext_sources_min": "EXT_SOURCES_MIN",
    "ext_sources_max": "EXT_SOURCES_MAX",
    "ext_sources_range": "EXT_SOURCES_RANGE",
    "enquiry_total": "AMT_REQ_CREDIT_BUREAU_TOTAL",
    "age_years": "AGE_YEARS",
    "down_payment": "DOWN_PAYMENT",
    "down_payment_ratio": "DOWN_PAYMENT_RATIO",
}

BASE_RATIO_FEATURES = [
    {"name": "DAYS_EMPLOYED_PERC", "numerator": "DAYS_EMPLOYED", "denominator": "DAYS_BIRTH"},
    {"name": "INCOME_PER_PERSON", "numerator": "AMT_INCOME_TOTAL", "denominator": "CNT_FAM_MEMBERS"},
]

GLOBAL_RATIO_FEATURES = [
    {"name": "CREDIT_INCOME_RATIO", "numerator": "AMT_CREDIT", "denominator": "AMT_INCOME_TOTAL"},
    {"name": "ANNUITY_INCOME_RATIO", "numerator": "AMT_ANNUITY", "denominator": "AMT_INCOME_TOTAL"},
    {"name": "CREDIT_TERM", "numerator": "AMT_ANNUITY", "denominator": "AMT_CREDIT"},
    {"name": "GOODS_CREDIT_RATIO", "numerator": "AMT_GOODS_PRICE", "denominator": "AMT_CREDIT"},
]

EXTENDED_RATIO_FEATURES = [
    {"name": "CREDIT_TO_ANNUITY_RATIO", "numerator": "AMT_CREDIT", "denominator": "AMT_ANNUITY"},
    {"name": "CREDIT_TO_GOODS_RATIO", "numerator": "AMT_CREDIT", "denominator": "AMT_GOODS_PRICE"},
    {"name": "REGISTRATION_BIRTH_RATIO", "numerator": "DAYS_REGISTRATION", "denominator": "DAYS_BIRTH"},
    {"name": "ID_PUBLISH_BIRTH_RATIO", "numerator": "DAYS_ID_PUBLISH", "denominator": "DAYS_BIRTH"},
    {"name": "PHONE_CHANGE_BIRTH_RATIO", "numerator": "DAYS_LAST_PHONE_CHANGE", "denominator": "DAYS_BIRTH"},
]

BUREAU_BALANCE = {
    "id_col": "SK_ID_BUREAU",
    "status_col": "STATUS",
    "month_col": "MONTHS_BALANCE",
}
BUREAU = {
    "prefix": "bureau_",
    "id_col": "SK_ID_BUREAU",
    "days_credit_col": "DAYS_CREDIT",
    "amt_credit_sum_col": "AMT_CREDIT_SUM",
    "amt_credit_sum_debt_col": "AMT_CREDIT_SUM_DEBT",
    "amt_credit_sum_limit_col": "AMT_CREDIT_SUM_LIMIT",
    "amt_credit_sum_overdue_col": "AMT_CREDIT_SUM_OVERDUE",
    "days_credit_enddate_col": "DAYS_CREDIT_ENDDATE",
    "days_enddate_fact_col": "DAYS_ENDDATE_FACT",
    "debt_credit_ratio_col": "DEBT_CREDIT_RATIO",
    "credit_active_col": "CREDIT_ACTIVE",
    "status_splits": {"active": "Active", "closed": "Closed"},
}
PREVIOUS_APPLICATION = {
    "prefix": "prev_",
    "id_col": "SK_ID_PREV",
    "amt_application_col": "AMT_APPLICATION",
    "amt_credit_col": "AMT_CREDIT",
    "app_credit_gap_col": "APP_CREDIT_GAP",
    "status_col": "NAME_CONTRACT_STATUS",
    "days_decision_col": "DAYS_DECISION",
    "amt_annuity_col": "AMT_ANNUITY",
    "cnt_payment_col": "CNT_PAYMENT",
    "total_payment_col": "PREV_TOTAL_PAYMENT",
    "simple_interest_col": "PREV_SIMPLE_INTEREST",
    "credit_to_annuity_col": "PREV_CREDIT_TO_ANNUITY",
    "status_values": {"approved": "Approved", "refused": "Refused", "canceled": "Canceled"},
}
POS_CASH = {
    "prefix": "pos_",
    "id_col": "SK_ID_PREV",
    "month_col": "MONTHS_BALANCE",
    "dpd_col": "SK_DPD",
    "cnt_instalment_col": "CNT_INSTALMENT",
    "cnt_instalment_future_col": "CNT_INSTALMENT_FUTURE",
    "cleanup_min_values": {"SK_DPD": 0, "CNT_INSTALMENT": 0, "CNT_INSTALMENT_FUTURE": 0},
}
INSTALLMENTS = {
    "prefix": "inst_",
    "instalment_version_col": "NUM_INSTALMENT_VERSION",
    "amt_payment_col": "AMT_PAYMENT",
    "amt_instalment_col": "AMT_INSTALMENT",
    "days_entry_payment_col": "DAYS_ENTRY_PAYMENT",
    "days_instalment_col": "DAYS_INSTALMENT",
    "payment_perc_col": "PAYMENT_PERC",
    "payment_diff_col": "PAYMENT_DIFF",
    "dpd_col": "DPD",
    "dbd_col": "DBD",
}
CREDIT_CARD_BALANCE = {
    "prefix": "cc_",
    "month_col": "MONTHS_BALANCE",
    "amt_balance_col": "AMT_BALANCE",
    "credit_limit_col": "AMT_CREDIT_LIMIT_ACTUAL",
    "drawings_col": "AMT_DRAWINGS_CURRENT",
    "dpd_col": "SK_DPD",
    "utilization_col": "UTILIZATION",
    "total_receivable_col": "AMT_TOTAL_RECEIVABLE",
    "receivable_principal_col": "AMT_RECEIVABLE_PRINCIPAL",
    "drawings_count_col": "CNT_DRAWINGS_CURRENT",
}
AGGREGATION_PREFIXES = [
    BUREAU["prefix"],
    PREVIOUS_APPLICATION["prefix"],
    POS_CASH["prefix"],
    INSTALLMENTS["prefix"],
    CREDIT_CARD_BALANCE["prefix"],
]
