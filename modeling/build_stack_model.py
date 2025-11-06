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


def get_oof_pre(xgb_params, lgbm_params, catboost_params, X, y, cv):
    xgb_pre = np.zeros(len(y))
    lgbm_pre = np.zeros(len(y))
    catboost_pre = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        xgb_fold = xgb.XGBRegressor(**xgb_params)
        xgb_fold.fit(X.iloc[train_idx], y.iloc[train_idx],eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],verbose=False)
        xgb_pre[val_idx] = xgb_fold.predict(X.iloc[val_idx])

        lgbm_fold = lgb.LGBMRegressor(**lgbm_params)
        lgbm_fold.fit(X.iloc[train_idx], y.iloc[train_idx],eval_set=[(X.iloc[val_idx], y.iloc[val_idx])])
        lgbm_pre[val_idx] = lgbm_fold.predict(X.iloc[val_idx])

        catboost_fold = cb.CatBoostRegressor(**catboost_params)
        catboost_fold.fit(X.iloc[train_idx], y.iloc[train_idx],eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],verbose=False)
        catboost_pre[val_idx] = catboost_fold.predict(X.iloc[val_idx])

    return xgb_pre, lgbm_pre, catboost_pre


#Meta model
def build_stacking_model(final_xgboost_params, final_lightgbm_params, final_catboost_params, 
                         X_train_transformed, y_train_transformed, preprocessor_y):
    
    #Lấy oof predict của 3 model để tiến hành tạo meta model(kết hợp 3 dự đoán)
    oof_pred_xgb, oof_pred_lgbm, oof_pred_cat = get_oof_pre(final_xgboost_params, 
                                                            final_lightgbm_params, 
                                                            final_catboost_params, 
                                                            X_train_transformed, 
                                                            y_train_transformed, cv)
    
    #Build data train meta model
    X_meta_train = pd.DataFrame({
        'meta_xgb': oof_pred_xgb,
        'meta_lgbm': oof_pred_lgbm,
        'meta_cat': oof_pred_cat
    })

    y_meta_train = y_train_transformed.copy()
    d_train_meta = xgb.DMatrix(X_meta_train, label=y_meta_train)

    #Dùng XGBoost để build model
    meta_model, final_meta_params = build_xgboost_model(d_train_meta, X_meta_train, y_meta_train, preprocessor_y)

    return meta_model, final_meta_params, X_meta_train