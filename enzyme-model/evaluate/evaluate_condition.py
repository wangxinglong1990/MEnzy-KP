"""Condition Model 评估模块。

提供回归指标 (R², MAE, RMSE) 以及 Pearson 相关系数。
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_regression(y_true, y_pred):
    """计算回归评估指标。

    Args:
        y_true: 真实值数组。
        y_pred: 预测值数组。

    Returns:
        dict: 包含 r2, mae, rmse, pearson 的字典。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Pearson correlation (handles constant arrays)
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        r_pearson, p_value = pearsonr(y_true, y_pred)
        r_pearson = float(r_pearson)
        p_value = float(p_value)
    else:
        r_pearson = 0.0
        p_value = 1.0

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": r_pearson,
        "pearson_p": p_value,
    }
