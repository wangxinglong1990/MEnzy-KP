"""评测框架：回归评估指标。"""

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_regression(y_true, y_pred):
    """计算回归评估指标。

    Args:
        y_true: 真实值数组
        y_pred: 预测值数组

    Returns:
        dict: 包含 r2, mae, rmse 的字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }
