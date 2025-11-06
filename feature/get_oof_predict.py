import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import os
from pandas.api.types import is_categorical_dtype, is_bool_dtype, is_float_dtype, is_integer_dtype


def generate_oof_elasticnet(X_train_df, y_train_df, X_test_df, cv, random_state=42):
    
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    oof_predcit_models = []

    X_train_arr = X_train_df.values
    y_train_arr = y_train_df.values.flatten()
    X_test_arr = X_test_df.values

    oof_predictions = np.zeros(X_train_arr.shape[0])
    test_predictions = np.zeros(X_test_arr.shape[0])

    scaler = StandardScaler()

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_arr, y_train_arr)):

        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state)

        X_train_fold, X_val_fold = X_train_arr[train_idx], X_train_arr[val_idx]
        y_train_fold = y_train_arr[train_idx]

        X_train_scaled = scaler.fit_transform(X_train_fold)

        X_val_scaled = scaler.transform(X_val_fold)
        X_test_scaled = scaler.transform(X_test_arr)

        model.fit(X_train_scaled, y_train_fold)
        oof_predcit_models.append({'fold': fold, 'model': model, 'scaler': scaler})

        oof_predictions[val_idx] = model.predict(X_val_scaled)

        test_predictions += model.predict(X_test_scaled) / cv.n_splits

    oof_col_name = 'OOF_ElasticNet'

    X_train_df[oof_col_name] = oof_predictions
    X_test_df[oof_col_name] = test_predictions

    return X_train_df, X_test_df, oof_predcit_models