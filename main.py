import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    # Fixed parameters across all experiments as requested
    args["tuned_epoch"] = 20
    args["optimizer"] = "sgd"
    args["init_lr"] = 0.05
    args["batch_size"] = 32

    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()