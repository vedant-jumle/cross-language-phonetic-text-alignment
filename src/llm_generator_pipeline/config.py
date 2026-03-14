import yaml

REQUIRED_KEYS = ["sample_size", "sample_seed", "llm_model", "llm_base_url", "batch_size", "n_perturbations", "target_scripts", "split_train", "split_val", "split_test", "hard_negative_warmup_steps", "hard_negative_mix_ratio"]

def load_config(config_path):
    with open(config_path, "rt", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if config is None:
        raise ValueError(f"Config file {config_path} is empty")
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config file {config_path} is missing required keys: {missing}")
    return config