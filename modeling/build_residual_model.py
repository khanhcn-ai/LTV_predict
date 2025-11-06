import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import logging
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.preprocessing import PowerTransformer, RobustScaler

from modeling.optuna_tunning import *
from modeling.build_base_model import *

eps = 1e-9
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def get_oof_res(meta_params, X, y, cv = cv):
    residuals = np.zeros(len(y))
    meta_pre = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        meta_fold = xgb.XGBRegressor(**meta_params)
        meta_fold.fit(X.iloc[train_idx], y.iloc[train_idx],eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],verbose=False)
        meta_pre[val_idx] = meta_fold.predict(X.iloc[val_idx])
        residuals[val_idx] = y.iloc[val_idx].values.flatten() 

    return residuals, meta_pre


def find_best_weight_residual(y_true_orig, yA_t, yBres_t, preY):
    def loss(w_arr):
        w = float(np.asarray(w_arr).item())
        y_pred_t = yA_t + w * yBres_t
        y_pred = preY.inverse_transform(y_pred_t.reshape(-1,1)).ravel()
        return nae(y_true_orig, y_pred)

    res = minimize(loss, x0=[0.5], bounds=[(0, 1)], method="L-BFGS-B")
    return float(res.x[0])


def build_res_model(final_meta_params, X_meta_train, y_train_transformed, preprocessor_y):

    #Tạo oof
    residuals, meta_oof = get_oof_res(final_meta_params, X_meta_train, y_train_transformed)
    X_res_train = X_meta_train.copy()

    #Tạo thêm các feature cho hậu chỉnh
    meta_cols = ['meta_xgb', 'meta_lgbm', 'meta_cat']
    X_res_train['meta_mean'] = X_res_train[meta_cols].mean(axis=1)
    X_res_train['meta_std'] = X_res_train[meta_cols].std(axis=1)
    X_res_train['range'] = X_res_train[meta_cols].max(axis=1) - X_res_train[meta_cols].min(axis=1)
    X_res_train['meta_oof'] = meta_oof
    y_res_train = y_train_transformed.copy()

    d_train_res = xgb.DMatrix(X_res_train, label=y_res_train)

    #Dùng XGBoost để build tiếp residual model
    res_model, final_res_params = build_xgboost_model(d_train_res, X_res_train, y_res_train, preprocessor_y)

    return res_model, final_res_params


def weight_optimize(xgboost_model, lightgbm_model, catboost_model,
                    meta_model, res_model, X_vali, y_vali, preprocessor_y):
    
    #Dùng các model để đưa ra dự đoán trên tập vali
    pred_xgb_vali = xgboost_model.predict(X_vali).flatten()
    pred_lgbm_vali = lightgbm_model.predict(X_vali).flatten()
    pred_catboost_vali = catboost_model.predict(X_vali).flatten()
    y_vali_orig = preprocessor_y.inverse_transform(y_vali.values.reshape(-1,1)).ravel()

    X_meta_vali = pd.DataFrame({
        'meta_xgb': pred_xgb_vali,
        'meta_lgbm': pred_lgbm_vali,
        'meta_cat': pred_catboost_vali
    })

    meta_cols = ['meta_xgb', 'meta_lgbm', 'meta_cat']
    X_meta_vali_base = np.column_stack([
        pred_xgb_vali,
        pred_lgbm_vali,
        pred_catboost_vali
    ])
    pred_meta_vali = meta_model.predict(X_meta_vali_base)
    meta_mean_vali = X_meta_vali[meta_cols].mean(axis=1)
    meta_std_vali = X_meta_vali[meta_cols].std(axis=1)
    meta_range_vali = X_meta_vali[meta_cols].max(axis=1) - X_meta_vali[meta_cols].min(axis=1)

    X_res_vali_final = np.column_stack([
        pred_xgb_vali,
        pred_lgbm_vali,
        pred_catboost_vali,
        meta_mean_vali,
        meta_std_vali,
        meta_range_vali,
        pred_meta_vali
    ])

    pred_res_vali = res_model.predict(X_res_vali_final)

    #Từ y_meta, y_res và y_true ta đi tìm w tối ưu cho y_true = y_meta + w * y_res
    w = find_best_weight_residual(y_vali_orig, pred_meta_vali, pred_res_vali, preprocessor_y)

    return w
    