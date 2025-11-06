import optuna
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
import logging
optuna.logging.set_verbosity(logging.WARNING) 


def nae_eval_metric(y_pred, y_true, preprocessor_y):
    y_true_transformed = y_true.get_label()
    y_pred_transformed = y_pred

    y_true_original = preprocessor_y.inverse_transform(
        y_true_transformed.reshape(-1, 1)
    ).flatten()

    y_pred_original = preprocessor_y.inverse_transform(
        y_pred_transformed.reshape(-1, 1)
    ).flatten()

    nae = np.mean(
        np.abs(y_true_original - y_pred_original) / (np.abs(y_true_original)+ 1e-9)
    )

    return 'custom_nae', nae * 100

def objective_xgboost_function(trial, d_train_xgboost, preprocessor_y):
    
    param = {
        'objective': 'reg:absoluteerror',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9)
    }

    cv_results = xgb.cv(
        params=param,
        dtrain=d_train_xgboost,
        num_boost_round=1000,
        nfold=5,
        custom_metric=lambda y_pred, y_true: nae_eval_metric(y_pred, y_true, preprocessor_y),
        maximize=False,
        as_pandas=True,
        early_stopping_rounds=30
    )
    best_iteration = cv_results['test-custom_nae-mean'].argmin()
    best_nae = cv_results['test-custom_nae-mean'].min()
    n_estimators_optimal = best_iteration + 1
    
    trial.set_user_attr('n_estimators_optimal', n_estimators_optimal)    
    return best_nae

def objective_lightgbm_function(trial, d_train_lightgbm):
    
    param = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'random_state': 42,
        'force_col_wise': True,
        'feature_pre_filter': False, 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 40),
        'max_depth': trial.suggest_int('max_depth', 3, 11),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1
    }
    
    cv_results = lgb.cv(
        params=param,
        train_set=d_train_lightgbm,
        num_boost_round=2000,
        nfold=5,
        stratified=False,
        callbacks=[lgb.early_stopping(30, verbose=False)],
        seed=42
    )
    
    best_iteration = cv_results['valid l1-mean'].index(min(cv_results['valid l1-mean']))
    best_mae = min(cv_results['valid l1-mean'])

    n_estimators_optimal = best_iteration + 1
    trial.set_user_attr('n_estimators_optimal', n_estimators_optimal) 
    
    return best_mae

def objective_catboost_function(trial, X_train_transformed, y_train_transformed):
    bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])

    param = {
        'loss_function': 'MAE',
        'eval_metric': "MAE",
        'random_seed': 42,
        'bootstrap_type': bootstrap_type,
        'verbose': -1,
        'task_type': 'GPU',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'depth': trial.suggest_int('max_depth', 3, 7),
        'l2_leaf_reg': trial.suggest_float('lambda', 1e-8, 100.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_child_weight', 3, 10),
    }

    if bootstrap_type == 'Bernoulli':
        param['subsample'] = trial.suggest_float('subsample', 0.5, 0.95)

    pool_cv = cb.Pool(
        data=X_train_transformed, 
        label=y_train_transformed
    )
    
    cv_results = cb.cv(
        pool=pool_cv,
        params=param,
        fold_count=5,
        iterations=1000,
        early_stopping_rounds=30,
        shuffle=True,
        verbose=False,
        seed=42
    )

    metric_key = 'test-MAE-mean'
    
    best_iteration = cv_results[metric_key].argmin()
    best_loss = cv_results[metric_key].min()
    n_estimators_optimal = best_iteration + 1
    
    trial.set_user_attr('n_estimators_optimal', n_estimators_optimal) 
    
    return best_loss
    