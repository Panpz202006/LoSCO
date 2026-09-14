import os
import copy
import json
import argparse

from trainer import start
from utils.toolkit import make_logdir
import ast


def main():
    # Parse arguments
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)   # Converting argparse Namespace to a dict
    args.update(param)  # Add parameters from json

    # Create log directory
    args['logdir'] = make_logdir(args)
    # config_path = os.path.join(args['logdir'], 'config.json')
    # with open(config_path, 'w') as f:
    #     json.dump(args, f, indent=4)

    # Set seed
    device = copy.deepcopy(args['device']).split(',')
    seed_list = copy.deepcopy(args['seed'])
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        start(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Reproduce of Multiple Incremental Learning Algorithms.',
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument('--config', type=str, default='./configs/baseline.json', help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--save_ckp', action='store_true', help='Whether to save checkpoints.')
    parser.add_argument('--moe_list', type=lambda x: ast.literal_eval(x), default=None, 
                        help='List of block indices to use MoE, e.g., "[8,9,10,11]"')
    return parser


if __name__ == '__main__':
    main()
