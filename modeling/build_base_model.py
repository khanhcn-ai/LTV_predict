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
eps = 1e-9
from modeling.optuna_tunning import *

def nae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))


#XGBoost
def build_xgboost_model(d_train, X_train, y_train, preprocessor_y):
    
    study_xgboost = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30))
    study_xgboost.optimize(
        lambda trial: objective_xgboost_function(trial, d_train, preprocessor_y),
        n_trials=30,
        show_progress_bar=True
    )

    #Lấy tham số khi tối ưu và fit
    best_xgboost_n_estimators = study_xgboost.best_trial.user_attrs.get('n_estimators_optimal') 
    final_xgboost_params = study_xgboost.best_params.copy()
    final_xgboost_params['n_estimators'] = best_xgboost_n_estimators
    final_xgboost_params['objective'] = 'reg:absoluteerror'
    final_xgboost_params['verbosity'] = 0
    best_xgb_model = xgb.XGBRegressor(**final_xgboost_params)

    best_xgb_model.fit(X_train, y_train)

    return best_xgb_model, final_xgboost_params


#LigthGBM
def build_lightgbm_model(d_train, X_train, y_train):
    
    study_lightgbm = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30))
    study_lightgbm.optimize(
        lambda trial: objective_lightgbm_function(trial, d_train),
        n_trials=30,
        show_progress_bar=True
    )

    #Lấy tham số khi tối ưu và fit
    best_lgbm_n_estimators = study_lightgbm.best_trial.user_attrs.get('n_estimators_optimal')
    final_lightgbm_params = study_lightgbm.best_params.copy()
    final_lightgbm_params['objective'] = 'regression_l1'
    final_lightgbm_params['n_estimators'] = best_lgbm_n_estimators
    final_lightgbm_params['metric'] = 'custom'
    final_lightgbm_params['random_state'] = 42
    final_lightgbm_params['boosting_type'] = 'gbdt'
    best_lgbm_model = lgb.LGBMRegressor(**final_lightgbm_params)

    best_lgbm_model.fit(X_train, y_train)

    return best_lgbm_model, final_lightgbm_params


#CatBoost
def build_catboost_model(X_train, y_train):
    
    study_catboost = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30))
    study_catboost.optimize(
        lambda trial: objective_catboost_function(trial, X_train, y_train),
        n_trials=30,
        show_progress_bar=True
    )

    #Lấy tham số khi tối ưu và fit
    best_catboost_iterations = study_catboost.best_trial.user_attrs.get('n_estimators_optimal')
    final_catboost_params = study_catboost.best_params.copy()
    final_catboost_params['depth'] = final_catboost_params.pop('max_depth')
    final_catboost_params['l2_leaf_reg'] = final_catboost_params.pop('lambda')
    final_catboost_params['min_data_in_leaf'] = final_catboost_params.pop('min_child_weight')
    final_catboost_params.pop('alpha', None)
    final_catboost_params['loss_function'] = 'MAE'
    final_catboost_params['eval_metric'] = 'MAE'
    final_catboost_params['iterations'] = best_catboost_iterations
    final_catboost_params['task_type'] = 'GPU'
    final_catboost_params['random_seed'] = 42
    final_catboost_params['verbose'] = 0
    best_catboost_model = cb.CatBoostRegressor(**final_catboost_params)

    best_catboost_model.fit(X_train, y_train)

    return best_catboost_model, final_catboost_params