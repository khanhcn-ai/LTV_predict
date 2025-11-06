import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import optuna
import catboost as cb
import joblib
import logging
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from sklearn.preprocessing import PowerTransformer, RobustScaler

NAE_OUTPUT_FILE = "./ans/NAE_ans_ensemble.txt"

#Thêm file
from feature.feature_adding import *
from feature.get_oof_predict import *
from feature.mixup_train_data import *
from modeling.build_base_model import *
from modeling.build_stack_model import *
from modeling.build_residual_model import *
from modeling.calculate_nae import *

eps = 1e-9
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def build_model_per_day(input_day):
    # Lấy data
    df = get_data_for_train(input_day)

    #Tạo dữ liệu để build
    features = df.drop(columns=[f'ltv_d{input_day}']).columns.tolist()
    target = f'ltv_d{input_day}'

    X = df[features]
    y = df[[target]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, poly_transformer = feature_for_X(X_train)
    X_test, _ = feature_for_X(X_test, poly_transformer)
    X_train, y_train = apply_mixup_train_data(X_train, y_train, augmentation_factor=1.0, random_state=42)
    features = X_train.columns.to_list()
    
    #Tranform dữ liệu
    preprocessor_X = PowerTransformer(method='yeo-johnson', standardize=False)
    preprocessor_y = PowerTransformer(method='yeo-johnson', standardize=False)
    
    X_train_transformed = pd.DataFrame(preprocessor_X.fit_transform(X_train), columns=features)
    X_test_transformed = pd.DataFrame(preprocessor_X.transform(X_test), columns=features)
    y_train_transformed = pd.DataFrame(preprocessor_y.fit_transform(y_train), columns=[f'ltv_d{input_day}'])
    y_test_transformed = pd.DataFrame(preprocessor_y.transform(y_test), columns=[f'ltv_d{input_day}'])

    #Tạo oof predict
    X_train_transformed, X_test_transformed, oof_predict_models = generate_oof_elasticnet(X_train_transformed, y_train_transformed, X_test_transformed, cv)
    features = X_train_transformed.columns.to_list()

    #Tạo DMatrix cho optuna
    d_train_xgboost = xgb.DMatrix(X_train_transformed, label=y_train_transformed)
    d_train_lightgbm = lgb.Dataset(X_train_transformed, label=y_train_transformed, params={'feature_pre_filter': False})

    #Chạy optuna

    #XGBoost
    print(f"Chạy Optuna XGBoost_base cho ngày thứ {input_day}")
    best_xgboost_model, final_xgboost_params = build_xgboost_model(d_train_xgboost, X_train_transformed, y_train_transformed, preprocessor_y)

    #LightGBM
    print(f"Chạy Optuna LightGBM_base cho ngày thứ {input_day}")
    best_lgbm_model, final_lightgbm_params = build_lightgbm_model(d_train_lightgbm, X_train_transformed, y_train_transformed)

    #CatBoost
    print(f"Chạy Optuna CatGBoost_base cho ngày thứ {input_day}")
    best_catboost_model, final_catboost_params = build_catboost_model(X_train_transformed, y_train_transformed)

    #Stacking model
    print(f"Chạy Optuna XGBoost_meta cho ngày thứ {input_day}")
    meta_model, final_meta_params, X_meta_train = build_stacking_model(final_xgboost_params, final_lightgbm_params, final_catboost_params, 
                                                                    X_train_transformed, y_train_transformed, preprocessor_y)

    #Build model hậu chỉnh
    print(f"Chạy Optuna XGBoost_residual cho ngày thứ {input_day}")
    res_model, final_res_params = build_res_model(final_meta_params, X_meta_train, y_train_transformed, preprocessor_y)

    #Tính nae
    nae = calculate_nae(best_xgboost_model, best_lgbm_model, best_catboost_model, 
                  meta_model, res_model, X_test_transformed, y_test_transformed, preprocessor_y)

    print(f"NAE tập test (Giá trị gốc) ngày {input_day}: {nae * 100:.4f}%")


    #Ghi kết quả vào file txt
    with open(NAE_OUTPUT_FILE, 'a', encoding='utf-8') as file:
        file.write(f"Giá trị NAE cho ngày {input_day} với ensemble là : {round(nae * 100, 4)}%." + "\n")


    #lưu lại model
    artifacts = {
        "power_X": preprocessor_X,
        "power_y": preprocessor_y,
        "feature_list": X_train_transformed.columns.tolist(),
        "poly_transform": poly_transformer,
        "oof_predict_models": oof_predict_models,
        "base_models": {
            "xgb": best_xgboost_model, "lgbm": best_lgbm_model, "cat": best_catboost_model
        },
        "randomstate" : 42,
        "meta_model": meta_model,
        "residual_model": res_model
    }
    joblib.dump(artifacts, f"./model/ltv_d{input_day}_stack_pipeline.joblib")

    print(f"Đã lưu thành công models cho ngày {input_day}")