"""配置加载器。

用法:
    from configs.config_loader import load_config
    cfg = load_config("configs/baseline/kcat.yaml")
"""

from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    """加载 YAML 配置文件并返回 dict。

    Args:
        path: YAML 配置文件路径。

    Returns:
        dict: 配置字典。
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"配置文件为空: {path}")
    return cfg
