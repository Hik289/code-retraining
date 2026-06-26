"""config.py — 模型配置加载

从 configs/*.yaml 读取模型配置，提供统一接口供所有 src/ 脚本使用。
"""
import os
import yaml

# configs/ 目录相对于本文件的位置
_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_model_config(model_name_or_path: str) -> dict:
    """加载模型配置。

    Args:
        model_name_or_path: 模型短名 (e.g. "santacoder") 或 YAML 文件路径。
            短名会自动映射到 configs/{name}.yaml。

    Returns:
        dict with all config fields from the YAML file.
    """
    if os.path.isfile(model_name_or_path):
        yaml_path = model_name_or_path
    else:
        yaml_path = os.path.join(_CONFIGS_DIR, f"{model_name_or_path}.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # 基本校验
    required = ["model_id", "short_name", "fim_prefix", "fim_middle", "fim_suffix"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config {yaml_path} missing required fields: {missing}")

    # 默认值
    cfg.setdefault("trust_remote_code", False)
    cfg.setdefault("fim_pad", None)
    cfg.setdefault("binary_good_token", " good")
    cfg.setdefault("binary_bad_token", " bad")
    cfg.setdefault("stop_sequences_humaneval", ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint"])
    cfg.setdefault("stop_sequences_mbpp", ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint"])

    return cfg


if __name__ == "__main__":
    # 快速验证：加载所有 4 个配置
    for name in ["santacoder", "starcoder2", "qwen25", "codellama"]:
        try:
            cfg = load_model_config(name)
            print(f"[OK] {name}: model_id={cfg['model_id']}, "
                  f"fim_prefix={cfg['fim_prefix']}, "
                  f"trust_remote_code={cfg['trust_remote_code']}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
