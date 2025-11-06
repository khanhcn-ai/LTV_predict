import joblib
import pandas as pd
import numpy as np
# Import tất cả các thư viện cần thiết mà models và preprocessors sử dụng
from sklearn.preprocessing import PowerTransformer 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# Import ElasticNet và StandardScaler nếu cần dùng các model trong oof_predict_models
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler 
import time
from feature.feature_adding import *

def nae_func(y_true, y_pred, eps=1e-9):
    return float(np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)))

def load_and_predict_ensemble(input_data_df, input_day, model_path="./model/"):
    start_time = time.time()
    file_name = f"ltv_d{input_day}_stack_pipeline.joblib"
    full_path = model_path + file_name

    # 1. Tải toàn bộ artifacts
    artifacts = joblib.load(full_path)

    # Trích xuất các thành phần
    power_X = artifacts['power_X']
    power_y = artifacts['power_y']
    poly_transformer = artifacts['poly_transform']
    xgb_model = artifacts['base_models']['xgb']
    lgbm_model = artifacts['base_models']['lgbm']
    cat_model = artifacts['base_models']['cat']
    feature_list = artifacts['feature_list']
    meta_model = artifacts['meta_model']
    residual_model = artifacts['residual_model']
    oof_predict_model = artifacts['oof_predict_models']

    df = input_data_df.copy()
    df = get_data_for_infer(df, input_day, poly_transformer)
    
    oof_col_name = 'OOF_ElasticNet'
    feature_list_base = [col for col in feature_list if col != oof_col_name]
    X_new = df[feature_list_base]
    y_new = df[f"ltv_d{input_day}"]
    # Áp dụng PowerTransformer đã fit (Power_X)
    X_new_transformed = pd.DataFrame(power_X.transform(X_new), columns=feature_list_base)

    #Tính oof predict
    X_new_arr = X_new.values
    test_oof_elasticnet = np.zeros(X_new_arr.shape[0])
    cv_n_splits = len(oof_predict_model)

    for item in oof_predict_model:
        model = item['model']
        scaler = item['scaler'] 
        
        X_new_scaled = scaler.transform(X_new_transformed.values)
        
        test_oof_elasticnet += model.predict(X_new_scaled) / cv_n_splits

    X_new_transformed[oof_col_name] = test_oof_elasticnet

    # --- 3. Dự đoán Stacking (3 bước) ---

    # 3.1. Dự đoán từ Base Models (Output Level-1)
    pred_xgb = xgb_model.predict(X_new_transformed).flatten()
    pred_lgbm = lgbm_model.predict(X_new_transformed).flatten()
    pred_cat = cat_model.predict(X_new_transformed).flatten()

    # Tạo input cho Meta và Residual Model
    X_meta_input = np.column_stack([pred_xgb, pred_lgbm, pred_cat])
    X_meta_temp = pd.DataFrame(X_meta_input, columns=['meta_xgb', 'meta_lgbm', 'meta_cat'])
    
    # 3.2. Dự đoán từ Meta Model
    pred_meta = meta_model.predict(X_meta_input)
    
    # 3.3. Dự đoán Residual Model (hậu chỉnh)
    meta_cols = ['meta_xgb', 'meta_lgbm', 'meta_cat']
    meta_mean = X_meta_temp[meta_cols].mean(axis=1)
    meta_std = X_meta_temp[meta_cols].std(axis=1)
    meta_range = X_meta_temp[meta_cols].max(axis=1) - X_meta_temp[meta_cols].min(axis=1)

    # Tạo features cho Residual Model
    X_res_input = np.column_stack([
        pred_xgb, pred_lgbm, pred_cat, 
        meta_mean, meta_std, meta_range, 
        pred_meta # Dự đoán Meta Model làm feature
    ])
    
    final_pred_transformed = residual_model.predict(X_res_input)
    
    # --- 4.Kết quả Cuối cùng ---

    final_prediction = power_y.inverse_transform(
        final_pred_transformed.reshape(-1, 1)
    ).flatten()

    nae_score = nae_func(y_new.values, final_prediction)
    end_time = time.time()
    elapsed_time = end_time - start_time

    return np.round(nae_score * 100, 4), np.round(elapsed_time, 4)


df_new = pd.read_csv("./data/2025-01-01_2025-03-02_puzzle_com.twisted.rope.tangle.csv")
OUTPUT_FILE = "./ans/ket_qua_danh_gia.txt"

def check_output(input_day):   
    nae, time = load_and_predict_ensemble(
        input_data_df=df_new, 
        input_day=input_day,
        model_path="./model/"
    )

    output_line = (
    f"Giá trị NAE cho d{input_day} : {nae:.4f}%\n"
    f"Thời gian dự đoán (trên 50 data mẫu): {time:.4f} giây\n"
    "-----------------------------------------\n"
    )

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as file:
        file.write(output_line)

    print(f"Đã ghi kết quả vào file: {OUTPUT_FILE}")


for i in range(4, 61, 2):
    check_output(i)