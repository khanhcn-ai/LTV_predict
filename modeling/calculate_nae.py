import numpy as np
import pandas as pd
import os
import sys
eps = 1e-9


def nae_func(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


def calculate_nae(xgboost_model, lightgbm_model, catboost_model, 
                  meta_model, res_model, X, y, preprocessor_y):
    pred_xgb_test = xgboost_model.predict(X).flatten()
    pred_lgbm_test = lightgbm_model.predict(X).flatten()
    pred_catboost_test = catboost_model.predict(X).flatten()

    X_meta_temp = pd.DataFrame({
        'meta_xgb': pred_xgb_test,
        'meta_lgbm': pred_lgbm_test,
        'meta_cat': pred_catboost_test
    })

    meta_cols = ['meta_xgb', 'meta_lgbm', 'meta_cat']
    X_meta_test_base = np.column_stack([
        pred_xgb_test,
        pred_lgbm_test,
        pred_catboost_test
    ])
    pred_meta_test = meta_model.predict(X_meta_test_base)
    meta_mean_test = X_meta_temp[meta_cols].mean(axis=1)
    meta_std_test = X_meta_temp[meta_cols].std(axis=1)
    meta_range_test = X_meta_temp[meta_cols].max(axis=1) - X_meta_temp[meta_cols].min(axis=1)

    X_res_test_final = np.column_stack([
        pred_xgb_test,
        pred_lgbm_test,
        pred_catboost_test,
        meta_mean_test,
        meta_std_test,
        meta_range_test,
        pred_meta_test
    ])

    pred_res = res_model.predict(X_res_test_final)
    final_pre = pred_res

    y_test_true = preprocessor_y.inverse_transform(
        y.values.reshape(-1, 1)
    ).flatten()

    final_prediction = preprocessor_y.inverse_transform(
        final_pre.reshape(-1, 1)
    ).flatten()

    nae = nae_func(y_test_true, final_prediction)

    return nae
    