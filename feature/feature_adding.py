import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import os
from pandas.api.types import is_categorical_dtype, is_bool_dtype, is_float_dtype, is_integer_dtype


base_dir = os.path.dirname(__file__)
data_file_name = "raw_2025-04-01_2025-05-31_puzzle_com.twisted.rope.tangle.csv"
data_folder = "data"
project_root = os.path.dirname(base_dir)
data_dir = os.path.join(project_root, data_folder, data_file_name)

data_original = pd.read_csv(data_dir)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def feature_for_X(df = data_original, poly_transformer=None):

    #Các feature
    df = df[['roas_d0','roas_d1','roas_d2','roas_d3',
            'cumulative_revenue_d0','cumulative_revenue_d1','cumulative_revenue_d2','cumulative_revenue_d3',
            'daily_revenue_d0','daily_revenue_d1','daily_revenue_d2',
            'unique_users_d0','unique_users_d1','unique_users_d2','unique_users_d3','daily_revenue_d3','cost',
            'ltv_d0', 'ltv_d1', 'ltv_d2', 'ltv_d3']].copy()
    df['ltv_mean'] = df[['ltv_d0', 'ltv_d1', 'ltv_d2', 'ltv_d3']].mean(axis=1)
    df['roas_mean'] = df[['roas_d0','roas_d1','roas_d2','roas_d3']].mean(axis=1)
    df['cumulative_revenue_mean'] = df[['cumulative_revenue_d0','cumulative_revenue_d1','cumulative_revenue_d2','cumulative_revenue_d3']].mean(axis=1)

    df['ltv_std'] = df[['ltv_d0', 'ltv_d1', 'ltv_d2', 'ltv_d3']].std(axis=1)
    df['roas_std'] = df[['roas_d0','roas_d1','roas_d2','roas_d3']].std(axis=1)
    df['cumulative_revenue_std'] = df[['cumulative_revenue_d0','cumulative_revenue_d1','cumulative_revenue_d2','cumulative_revenue_d3']].std(axis=1)

    df['ltv_growth'] = (df['ltv_d3'] - df['ltv_d0']) / (3 + 1e-9)
    df['cumulative_revenue_growth'] = df['cumulative_revenue_d3'] - df['cumulative_revenue_d0']

    df['revenue_acceleration'] = df['daily_revenue_d3'] - df['daily_revenue_d2'] - df['daily_revenue_d1'] + df['daily_revenue_d0']
    df['user_acceleration'] = df['unique_users_d3'] - df['unique_users_d2'] - df['unique_users_d1'] + df['unique_users_d0']

    df['roas_trend'] = df['roas_d3'] - df['roas_d0']
    df['ltv_roas_ratio'] = df['ltv_d3'] / df['roas_d3']

    df['ltv_slope_d0_d1'] = df['ltv_d1'] - df['ltv_d0']
    df['ltv_slope_d1_d2'] = df['ltv_d2'] - df['ltv_d1']
    df['ltv_slope_d2_d3'] = df['ltv_d3'] - df['ltv_d2']
    df['ARPU_d0'] = df['daily_revenue_d0'] / df['unique_users_d0']
    df['ARPU_d1'] = df['daily_revenue_d1'] / df['unique_users_d1']
    df['ARPU_d2'] = df['daily_revenue_d2'] / df['unique_users_d2']
    df['ARPU_d3'] = df['daily_revenue_d3'] / df['unique_users_d3']
    df['retention_d1'] = df['unique_users_d1'] / df['unique_users_d0']
    df['retention_d2'] = df['unique_users_d2'] / df['unique_users_d0']
    df['retention_d3'] = df['unique_users_d3'] / df['unique_users_d0']

    df['ltv_acceleration'] = df['ltv_slope_d2_d3'] - df['ltv_slope_d1_d2']
    df['roas_slope_d0_d1'] = df['roas_d1'] - df['roas_d0']
    df['roas_slope_d1_d2'] = df['roas_d2'] - df['roas_d1']
    df['roas_slope_d2_d3'] = df['roas_d3'] - df['roas_d2']
    df['roas_acceleration'] = df['roas_slope_d2_d3'] - df['roas_slope_d1_d2']
    df['is_ltv_slowing_down'] = (df['ltv_acceleration'] < 0).astype(int)
    df['is_roas_slowing_down'] = (df['roas_acceleration'] < 0).astype(int)

    df['ltv_gain'] = df['ltv_d3'] - df['ltv_d0']
    df['cumulative_users_d3'] = df['unique_users_d0'] + df['unique_users_d1'] + df['unique_users_d2'] + df['unique_users_d3']
    df['ARPU_cumulative_d3'] = df['cumulative_revenue_d3'] / df['cumulative_users_d3']
    df['ARPU_trend'] = (df['daily_revenue_d3'] / (df['unique_users_d3'] + 1e-9)) - (df['daily_revenue_d0'] / (df['unique_users_d0'] + 1e-9))
    df['Payback_Velocity'] = (df['cumulative_revenue_d3'] / df['cost']) / 4
    df['Acceleration_Ratio'] = df['revenue_acceleration'] / (df['user_acceleration'] + 1e-9)
    df['CAC'] = df['cost'] / df['cumulative_users_d3']
    df['ROAS_CV'] = df['roas_std'] / (df['roas_mean'] + 1e-9)
    df['ERTI'] = df['cumulative_users_d3'] / df['cost']
    df['LTV_CAC'] = df['ltv_d3'] / df['CAC']

    df['daily_to_cumulative_revenue_ratio_d3'] = df['daily_revenue_d3'] / (df['cumulative_revenue_d3'] + 1e-9)
    df['d0_cohort_value_d3'] = df['cumulative_revenue_d3'] / (df['unique_users_d0'] + 1e-9)
    df['user_growth_d3_vs_d0'] = df['unique_users_d3'] / (df['unique_users_d0'] + 1e-9)
    df['cost_per_revenue_d3'] = df['cost'] / (df['cumulative_revenue_d3'] + 1e-9)
    df['arpu_d3_x_payback'] = df['ARPU_cumulative_d3'] * df['Payback_Velocity']
    df['LTV_CV'] = df['ltv_std'] / (df['ltv_mean'] + 1e-9)
    df['LTV_CAC_Trend'] = df['LTV_CAC'] * df['ARPU_trend']
    df['Cost_LTV_Mean'] = df['cost'] * df['ltv_mean']
    df['Payback_Accel'] = df['Payback_Velocity'] * df['ltv_acceleration']
    df['ROAS_Std_Weighted'] = df['roas_std'] * df['LTV_CAC']
    df['Daily_to_Cumul_LTV'] = df['daily_to_cumulative_revenue_ratio_d3'] * df['ltv_d3']

    power_cols_base = ['ltv_mean', 'cost', 'LTV_CAC', 'Payback_Velocity', 'ltv_acceleration', 'roas_mean','ARPU_trend']
    X_poly_base = df[power_cols_base]

    if poly_transformer == None:
        poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
        X_poly_transformed = poly_transformer.fit_transform(X_poly_base)
    else:
        X_poly_transformed = poly_transformer.transform(X_poly_base)

    new_poly_names = poly_transformer.get_feature_names_out(input_features=power_cols_base)
    df_poly = pd.DataFrame(X_poly_transformed, columns=new_poly_names, index=df.index)
    df = pd.concat([df.drop(columns=power_cols_base), df_poly], axis=1)
    df = df.fillna(0)

    df = df.drop(columns=['daily_revenue_d0', 'LTV_CAC ARPU_trend', 'ltv_mean Payback_Velocity', 'LTV_CAC roas_mean', 'LTV_CAC Payback_Velocity', 'roas_mean',
                         'ARPU_cumulative_d3', 'cost Payback_Velocity', 'cost', 'revenue_acceleration', 'LTV_CV', 'Payback_Velocity roas_mean', 'Payback_Velocity',
                         'cumulative_revenue_d2', 'cumulative_revenue_growth', 'Payback_Velocity ARPU_trend', 'LTV_CAC^2', 'cumulative_users_d3', 'cumulative_revenue_mean',
                         'Payback_Velocity ltv_acceleration', 'roas_mean^2', 'cost roas_mean', 'cumulative_revenue_std', 'cost_per_revenue_d3', 'cumulative_revenue_d3',
                         'ltv_mean cost', 'ltv_acceleration roas_mean', 'cumulative_revenue_d1', 'cost^2', 'is_ltv_slowing_down', 'LTV_CAC_Trend', 'Payback_Velocity^2',
                         'is_roas_slowing_down', 'arpu_d3_x_payback', 'daily_revenue_d2', 'daily_revenue_d3', 'roas_mean ARPU_trend', 'cost LTV_CAC', 'LTV_CAC ltv_acceleration', 'Payback_Accel', 'roas_trend', 'ROAS_Std_Weighted'])#

    return df, poly_transformer


def get_data_for_train(input_day):
    
    df = data_original.copy()

    df = df[['roas_d0','roas_d1','roas_d2','roas_d3',
            'cumulative_revenue_d0','cumulative_revenue_d1','cumulative_revenue_d2','cumulative_revenue_d3',
            'daily_revenue_d0','daily_revenue_d1','daily_revenue_d2',
            'unique_users_d0','unique_users_d1','unique_users_d2','unique_users_d3','daily_revenue_d3','cost',
            'ltv_d0', 'ltv_d1', 'ltv_d2', 'ltv_d3', f'ltv_d{input_day}']].copy()
    
    return df

def get_data_for_infer(df_new, input_day, poly_transformer):

    df = df_new.copy()

    df = df[['roas_d0','roas_d1','roas_d2','roas_d3',
                'cumulative_revenue_d0','cumulative_revenue_d1','cumulative_revenue_d2','cumulative_revenue_d3',
                'daily_revenue_d0','daily_revenue_d1','daily_revenue_d2',
                'unique_users_d0','unique_users_d1','unique_users_d2','unique_users_d3','daily_revenue_d3','cost',
                'ltv_d0', 'ltv_d1', 'ltv_d2', 'ltv_d3', f'ltv_d{input_day}']].copy()

    features = df.drop(columns=[f'ltv_d{input_day}']).columns.tolist()
    target = f'ltv_d{input_day}'

    X = df[features]
    y = df[[target]]

    X, _ = feature_for_X(X, poly_transformer)
    ans = pd.concat([X, y], axis=1)
    
    return ans


