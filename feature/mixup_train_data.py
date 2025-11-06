import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import os
from pandas.api.types import is_categorical_dtype, is_bool_dtype, is_float_dtype, is_integer_dtype


def apply_mixup_train_data(X_train_df, y_train_df, augmentation_factor=1.0, random_state=42):
    if augmentation_factor <= 0:
        return X_train_df.copy(), y_train_df.copy()

    X = X_train_df.reset_index(drop=True)
    y = y_train_df.reset_index(drop=True)
    N = len(X)
    if N == 0:
        return X.copy(), y.copy()

    N_aug = int(N * augmentation_factor)
    if N_aug <= 0:
        return X.copy(), y.copy()

    ohe_groups = []
    ohe_cols = set()

    rng = np.random.default_rng(random_state)
    idx_A = rng.integers(0, N, size=N_aug)
    idx_B = (idx_A + (rng.integers(1, N, size=N_aug) if N > 1 else 0)) % max(N, 1)
    lam = rng.beta(1.0, 1.0, size=N_aug).reshape(-1, 1)
    pickA = (lam.ravel() > 0.5)

    cols = list(X.columns)
    
    float_cols = [c for c in cols if is_float_dtype(X[c].dtype)] 
    int_cols = [c for c in cols if is_integer_dtype(X[c].dtype)] 
    bool_cols = [c for c in cols if is_bool_dtype(X[c].dtype)]
    
    int_cols = [c for c in int_cols if c not in bool_cols and c not in float_cols]
    
    handled_cols = set(float_cols) | set(int_cols) | set(bool_cols) 
    other_cols = [c for c in cols if c not in handled_cols]

    new_cols = {}

    if float_cols:
        A = X[float_cols].to_numpy()[idx_A]
        B = X[float_cols].to_numpy()[idx_B]
        M = lam * A + (1.0 - lam) * B
        for j, c in enumerate(float_cols):
            new_cols[c] = M[:, j].astype(X[c].dtype, copy=False)

    if int_cols:
        A = X[int_cols].astype('float64', copy=False).to_numpy()[idx_A]
        B = X[int_cols].astype('float64', copy=False).to_numpy()[idx_B]
        M = lam * A + (1.0 - lam) * B
        R = np.rint(M)
        mins = X[int_cols].astype('float64', copy=False).min().to_numpy()
        maxs = X[int_cols].astype('float64', copy=False).max().to_numpy()
        C = np.clip(R, mins, maxs)
        for j, c in enumerate(int_cols):
            new_cols[c] = C[:, j].astype(X[c].dtype, copy=False)

    if bool_cols:
        A = X[bool_cols].to_numpy()[idx_A]
        B = X[bool_cols].to_numpy()[idx_B]
        M = np.where(pickA[:, None], A, B)
        for j, c in enumerate(bool_cols):
            new_cols[c] = M[:, j].astype(X[c].dtype, copy=False)

    if other_cols:
        A = X[other_cols].to_numpy(dtype=object)[idx_A]
        B = X[other_cols].to_numpy(dtype=object)[idx_B]
        M = np.where(pickA[:, None], A, B)
        for j, c in enumerate(other_cols):
            new_cols[c] = M[:, j]

    X_new = pd.DataFrame({c: new_cols[c] for c in cols}, columns=cols)

    yA = y.to_numpy(dtype=np.float64)[idx_A]
    yB = y.to_numpy(dtype=np.float64)[idx_B]
    y_new = lam * yA + (1.0 - lam) * yB
    y_new_df = pd.DataFrame(y_new, columns=y.columns)

    X_aug = pd.concat([X, X_new], ignore_index=True)
    y_aug = pd.concat([y, y_new_df], ignore_index=True)
    return X_aug, y_aug
