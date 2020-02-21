import os

from .. import ExperimentBuilder
from ..argparse_utils import add_exp_arguments

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Remove unnecessary weights. Remove everything except the best model and the last model.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_exp_arguments(parser, dataset_choices=None)

    return parser

parser = get_parser()
args = parser.parse_args()


if __name__ == '__main__':
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.experiment_name)
    exp.load_summary()

    best_epoch = exp.get_best_model_stat()['epoch']
    last_epoch = max(exp.summary['epoch'])
    remove_list = []
    keep_list = []

    print("These files will be removed:")
    #best_stat['epoch'], exp.summary['epoch'][-1]
    for epoch in exp.summary['epoch']:
        checkpoint_path = exp.get_checkpoint_path(epoch)
        if epoch not in [best_epoch, last_epoch]:
            remove_list.append(checkpoint_path)
            print(checkpoint_path)
        else:
            keep_list.append((checkpoint_path, exp.get_epoch_stat(epoch)))

    print()
    print("These files will be kept:")
    for checkpoint_path, stat in keep_list:
        print(checkpoint_path, stat)

    proceed = input("Proceed? (y/n) ")
    if proceed.lower() == 'y':
        print("Removing..")
        for remove_path in remove_list:
            os.remove(remove_path)


    print("Successfully removed.")
    
