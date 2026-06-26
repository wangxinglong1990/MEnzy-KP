"""Compare Models — 模型对比工具.

提供模型间指标对比和排名功能。
"""


def compare_models(results):
    """对比多个模型的评估指标，输出排名表。

    Args:
        results: dict, 格式如 {"model_name": {"r2": 0.67, "mae": 0.60, "rmse": 0.88}}

    Returns:
        list[dict]: 按 R² 降序排列的排名列表。
    """
    if not results:
        return []

    ranked = []
    for name, metrics in results.items():
        ranked.append({
            "model": name,
            "r2": metrics.get("r2", float("-inf")),
            "mae": metrics.get("mae", float("inf")),
            "rmse": metrics.get("rmse", float("inf")),
        })

    ranked.sort(key=lambda x: x["r2"], reverse=True)

    print(f"{'Rank':<6} {'Model':<12} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 48)
    for i, entry in enumerate(ranked, 1):
        print(f"{i:<6} {entry['model']:<12} {entry['r2']:<10.4f} {entry['mae']:<10.4f} {entry['rmse']:<10.4f}")

    return ranked
