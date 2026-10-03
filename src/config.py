import os
import yaml

_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_model_config(model_name_or_path: str) -> dict:
    if os.path.isfile(model_name_or_path):
        yaml_path = model_name_or_path
    else:
        yaml_path = os.path.join(_CONFIGS_DIR, f"{model_name_or_path}.yaml")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    required = ["model_id", "short_name", "fim_prefix", "fim_middle", "fim_suffix"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config {yaml_path} missing required fields: {missing}")

    cfg.setdefault("trust_remote_code", False)
    cfg.setdefault("fim_pad", None)
    cfg.setdefault("binary_good_token", " good")
    cfg.setdefault("binary_bad_token", " bad")
    cfg.setdefault("stop_sequences_humaneval", ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint"])
    cfg.setdefault("stop_sequences_mbpp", ["\nclass ", "\ndef ", "\n#", "\nif ", "\nprint"])

    return cfg


if __name__ == "__main__":
    for name in ["santacoder", "starcoder2", "qwen25", "codellama"]:
        try:
            cfg = load_model_config(name)
            print(f"[OK] {name}: model_id={cfg['model_id']}, "
                  f"fim_prefix={cfg['fim_prefix']}, "
                  f"trust_remote_code={cfg['trust_remote_code']}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
