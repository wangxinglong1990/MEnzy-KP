"""StackingV2 评估模块。

支持 R², MAE, RMSE, Pearson, Spearman。
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def evaluate_stacking(y_true, y_pred):
    """计算回归评估指标，含 Spearman 秩相关。

    Args:
        y_true: 真实值数组。
        y_pred: 预测值数组。

    Returns:
        dict: 包含 r2, mae, rmse, pearson_r, pearson_p, spearman_r, spearman_p。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Pearson
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pr, pp = pearsonr(y_true, y_pred)
        pearson_r = float(pr)
        pearson_p = float(pp)
    else:
        pearson_r, pearson_p = 0.0, 1.0

    # Spearman
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        sr, sp = spearmanr(y_true, y_pred)
        spearman_r = float(sr)
        spearman_p = float(sp)
    else:
        spearman_r, spearman_p = 0.0, 1.0

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }
