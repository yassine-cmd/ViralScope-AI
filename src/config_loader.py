"""ConfigLoader — Load & validate config.yaml, create directories."""

import os


class ConfigLoader:
    """Load and validate configuration from config.yaml."""

    REQUIRED_KEYS = {
        "project": ["name", "seed", "device"],
        "data": [
            "raw_dir", "processed_dir", "tensor_dir",
            "train_split", "val_split", "test_split",
            "min_dataset_size", "max_per_class", "buffer_multiplier",
            "target_threshold", "sampling_strategy",
            "thumbnail_url_template", "thumbnail_fallback_url",
            "thumbnail_rate_limit", "thumbnail_download_workers",
        ],
        "model": ["type", "embedding"],
        "model.embedding": ["checkpoint", "image_dim", "text_dim", "max_seq_length"],
        "model.probe_pairs": [],
        "paths": ["best_model", "training_log", "results"],
    }

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = None
        self.failed_video_ids = []

    def load(self):
        """Load config from YAML file."""
        import yaml

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"{self.config_path} not found in {os.getcwd()}. "
                "Place config.yaml in the working directory before continuing."
            )
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        if not self.config:
            raise ValueError("config.yaml is empty — provide a valid configuration file.")
        return self

    def validate(self):
        """Validate all required keys are present. Raises KeyError with details if missing."""
        missing = []
        for section, keys in self.REQUIRED_KEYS.items():
            if "." in section:
                top, sub = section.split(".", 1)
                node = self.config.get(top, {}).get(sub)
                if node is None:
                    missing.append(f"  CONFIG['{top}']['{sub}'] block is missing")
                    continue
                for key in keys:
                    if key not in node:
                        missing.append(f"  CONFIG['{top}']['{sub}']['{key}'] is missing")
            else:
                if section not in self.config:
                    missing.append(f"  CONFIG['{section}'] block is missing")
                    continue
                for key in keys:
                    if key not in self.config[section]:
                        missing.append(f"  CONFIG['{section}']['{key}'] is missing")
        if missing:
            raise KeyError("Config validation failed — missing keys:\n" + "\n".join(missing))
        return self

    def create_directories(self):
        """Create all required output directories."""
        dirs = [
            self.config["data"]["raw_dir"],
            f"{self.config['data']['raw_dir']}/thumbnails",
            self.config["data"]["processed_dir"],
            self.config["data"]["tensor_dir"],
            os.path.dirname(self.config["paths"]["best_model"]),
            os.path.dirname(self.config["paths"]["training_log"]),
            self.config["paths"]["results"],
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        return self

    def print_summary(self):
        """Print configuration summary."""
        cfg = self.config
        print(f"[Config] Loaded and validated : {self.config_path}")
        print(f"[Config] Project              : {cfg['project']['name']}")
        print(f"[Config] Seed                 : {cfg['project']['seed']}")
        print(f"[Config] Backbone             : {cfg['model']['embedding']['checkpoint']}")
        print(f"[Config] Embedding dims       : img={cfg['model']['embedding']['image_dim']}  "
              f"txt={cfg['model']['embedding']['text_dim']}")
        print(f"[Config] max_per_class        : {cfg['data'].get('max_per_class', 999999):,}")
        print(f"[Config] sampling_strategy    : {cfg['data']['sampling_strategy']}")
        return self
