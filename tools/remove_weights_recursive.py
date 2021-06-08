import os

from experiment_utils import ExperimentBuilder
from experiment_utils.argparse_utils import add_exp_arguments

from remove_weights import remove_weights

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Remove unnecessary weights. Remove everything except the best model and the last model.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-R", "--experiment_root", type=str, help="Experiment output root directory")

    return parser

parser = get_parser()
args = parser.parse_args()
    
if __name__ == '__main__':
    datasets = sorted(os.listdir(args.experiment_root))
    for dataset in datasets:
        models = sorted(os.listdir(os.path.join(args.experiment_root, dataset)))
        for model in models:
            experiments = sorted(os.listdir(os.path.join(args.experiment_root, dataset, model)))
            for experiment in experiments:
                print(dataset, model, experiment)
                remove_weights(args.experiment_root, dataset, model, experiment)
