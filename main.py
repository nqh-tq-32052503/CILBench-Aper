import json
import argparse
import copy
from trainer import train

def main():
    parser = setup_parser()
    cli_args = parser.parse_args()
    raw_config = load_json(cli_args.config)

    # Support scenario config format (containing a "models" list), a list of configs, or a single config
    configs_to_run = []
    if isinstance(raw_config, list):
        for item in raw_config:
            configs_to_run.append(copy.deepcopy(item))
    elif isinstance(raw_config, dict) and "models" in raw_config:
        common_cfg = {k: v for k, v in raw_config.items() if k != "models"}
        for model_item in raw_config["models"]:
            cfg = copy.deepcopy(common_cfg)
            cfg.update(model_item)
            configs_to_run.append(cfg)
    else:
        configs_to_run.append(copy.deepcopy(raw_config))

    # Apply CLI overrides and fixed parameters across all experiments
    final_configs = []
    for cfg in configs_to_run:
        if cli_args.device is not None:
            cfg["device"] = [str(cli_args.device)]
        if cli_args.seed is not None:
            cfg["seed"] = [int(s) for s in cli_args.seed.split(",")]

        # Fixed parameters across all experiments as requested
        cfg["tuned_epoch"] = 20
        cfg["optimizer"] = "sgd"
        cfg["init_lr"] = 0.05
        cfg["batch_size"] = 32

        # Filter by model/prefix if requested
        if cli_args.model is not None:
            if cfg.get("model_name") != cli_args.model and cfg.get("prefix") != cli_args.model:
                continue

        final_configs.append(cfg)

    print(f"Total experiment configurations to run: {len(final_configs)}")
    for idx, cfg in enumerate(final_configs, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(final_configs)}] Running: Model={cfg.get('model_name')} ({cfg.get('prefix')}) | "
              f"Dataset={cfg.get('dataset')} | Tasks={cfg.get('init_cls')}+{cfg.get('increment')} | Seeds={cfg.get('seed')}")
        print(f"{'='*70}\n")
        train(cfg)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings or scenario.')
    parser.add_argument('--model', type=str, default=None,
                        help='Optional filter to run only a specific model or prefix (e.g., aper_ssf, exp_vpt-deep).')
    parser.add_argument('--device', type=str, default=None,
                        help='Optional override for GPU device index (e.g., 0).')
    parser.add_argument('--seed', type=str, default=None,
                        help='Optional override for seeds separated by comma (e.g., 1993,1994,1995).')
    return parser

if __name__ == '__main__':
    main()