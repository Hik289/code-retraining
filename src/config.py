import copy
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


EXPERIMENT_CONFIG = {
    'defaults': {
        'rounds': 5,
        'seeds': [0],
        'accepted_samples': 5000,
        'raw_samples': 20000,
        'retained_fraction': 0.25,
        'candidate_batch': 2000,
        'max_candidates': 200000,
        'prompt_tokens': 1024,
        'max_new_tokens': 1024,
        'generation_batch_size': 8,
        'score_batch_size': 8,
        'temperature': 0.8,
        'top_p': 0.95,
        'repetition_threshold': 0.3,
        'min_completion_tokens': 50,
        'prompt_seed': 0,
        'training': {
            'max_steps': 3000,
            'seq_length': 2048,
            'learning_rate': 5e-05,
            'weight_decay': 0.05,
            'warmup_steps': 500,
            'fim_rate': 0.0,
        },
        'evaluation': {
            'datasets': ['humaneval', 'mbpp', 'livecodebench'],
            'n_samples': 200,
            'temperature': 0.8,
            'top_p': 0.95,
            'max_new_tokens': 512,
            'release_version': 'release_v1',
            'workers': 16,
        },
        'execution': {'samples': 500, 'timeout': 5, 'workers': 4, 'image': 'python:3.11-slim'},
    },
    'models': {
        'santacoder': {
            'config': 'configs/santacoder.yaml',
            'nl_fraction': 0.0,
            'batch_size': 32,
            'gradient_accumulation_steps': 2,
        },
        'starcoder2': {
            'config': 'configs/starcoder2.yaml',
            'nl_fraction': 0.25,
            'batch_size': 8,
            'gradient_accumulation_steps': 8,
        },
        'qwen25': {
            'config': 'configs/qwen25.yaml',
            'nl_fraction': 0.25,
            'batch_size': 16,
            'gradient_accumulation_steps': 4,
        },
        'codellama': {
            'config': 'configs/codellama.yaml',
            'nl_fraction': 0.0,
            'batch_size': 8,
            'gradient_accumulation_steps': 8,
        },
    },
    'suites': {
        'main': {
            'models': ['santacoder', 'starcoder2', 'qwen25', 'codellama'],
            'gates': ['none', 'compile', 'quality', 'ppl', 'binary'],
        },
        'execution_review': {'models': ['qwen25'], 'gates': ['none', 'compile', 'binary', 'execution']},
        'anchor': {
            'models': ['qwen25'],
            'nl_fraction': 0.0,
            'cases': [
                {'name': 'synthetic_only', 'gate': 'none', 'anchor_fraction': 0.0},
                {'name': 'execution_only', 'gate': 'execution', 'anchor_fraction': 0.0},
                {'name': 'execution_anchor_5pct', 'gate': 'execution', 'anchor_fraction': 0.05},
            ],
        },
        'noisy_gate': {
            'models': ['qwen25'],
            'cases': [
                {'name': 'compile_noise_0', 'gate': 'compile', 'noise': 0.0},
                {'name': 'compile_noise_5pct', 'gate': 'compile', 'noise': 0.05},
                {'name': 'compile_noise_10pct', 'gate': 'compile', 'noise': 0.1},
                {'name': 'compile_noise_20pct', 'gate': 'compile', 'noise': 0.2},
            ],
        },
        'single_seed': {'models': ['santacoder', 'qwen25'], 'gates': ['none', 'compile', 'binary']},
        'extended': {
            'models': ['santacoder'],
            'cases': [
                {'name': 'none_10r', 'gate': 'none', 'rounds': 10},
                {'name': 'compile_10r', 'gate': 'compile', 'rounds': 10},
                {'name': 'compile_quality_10r', 'gate': 'compile+quality', 'rounds': 10},
                {'name': 'ppl_10r', 'gate': 'ppl', 'rounds': 10},
                {'name': 'binary_10r', 'gate': 'binary', 'rounds': 10},
                {
                    'name': 'quality_20r',
                    'gate': 'quality',
                    'rounds': 20,
                    'evaluation_rounds': [0, 1, 2, 3, 4, 5, 10, 15, 20],
                },
            ],
        },
    },
}


def load_experiment_config(path=None):
    if path is None:
        return copy.deepcopy(EXPERIMENT_CONFIG)
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


if __name__ == "__main__":
    for name in ["santacoder", "starcoder2", "qwen25", "codellama"]:
        try:
            cfg = load_model_config(name)
            print(f"[OK] {name}: model_id={cfg['model_id']}, "
                  f"fim_prefix={cfg['fim_prefix']}, "
                  f"trust_remote_code={cfg['trust_remote_code']}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
