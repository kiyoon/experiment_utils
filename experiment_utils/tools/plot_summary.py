#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys, os
import warnings

from .. import ExperimentBuilder


warnings.filterwarnings('error')
plt.rcParams.update({'font.size': 16})

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Plot multiple experiment into one graph to compare the figures.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--experiment_root", required=True, help="Experiment root directory without experiment names.")
    parser.add_argument("-d", "--dataset", required=True, help="Name of the dataset.")
    parser.add_argument("-e", "--experiment_name", required=True, help="Name of experiment to plot.")
    return parser

parser = get_parser()
args = parser.parse_args()


if __name__ == '__main__':
    exp = ExperimentBuilder(args.experiment_root, args.dataset, args.experiment_name)
    exp.load_summary()
    exp.plot_summary()
