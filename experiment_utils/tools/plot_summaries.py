#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys, os
import warnings


warnings.filterwarnings('error')
plt.rcParams.update({'font.size': 16})

import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Plot multiple experiment into one graph to compare the figures.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--experiment_root", required=True, help="Experiment root directory without experiment names.")
    parser.add_argument("-d", "--dataset", required=True, help="Name of the dataset.")
    parser.add_argument("-e", "--experiment_names", required=True, nargs='+', help="Names of experiments to plot.")
    return parser

parser = get_parser()
args = parser.parse_args()

def plot_multiple_stats(experiment_root, dataset, experiment_names):
    output_fig_dir = os.path.join(experiment_root, dataset, 'comparisons', ','.join(experiment_names))
    os.makedirs(output_fig_dir, exist_ok=True)

    label = {'train_loss': 'training loss', 'train_acc': 'training accuracy', 'val_loss': 'validation clip loss', 'val_acc': 'validation clip accuracy', 'val_vid_acc_top1': 'validation video accuracy top-1', 'val_vid_acc_top5': 'validation video accuracy top-5'}

    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    ax_1.set_xlim(auto=True)
    ax_1.set_ylim(auto=True)
    ax_1.set_xlabel('Epoch number')
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    ax_2.set_xlim(auto=True)
    ax_2.set_ylim(auto=True)
    ax_2.set_xlabel('Epoch number')
    fig_3 = plt.figure(figsize=(8, 4))
    ax_3 = fig_3.add_subplot(111)
    ax_3.set_xlim(auto=True)
    ax_3.set_ylim(auto=True)
    ax_3.set_xlabel('Epoch number')
    fig_4 = plt.figure(figsize=(8, 4))
    ax_4 = fig_4.add_subplot(111)
    ax_4.set_xlim(auto=True)
    ax_4.set_ylim(auto=True)
    ax_4.set_xlabel('Epoch number')

    for experiment_name in experiment_names:
        stats_file = os.path.join(experiment_root, dataset, experiment_name, 'logs', 'summary.csv')
        stats = np.genfromtxt(stats_file, delimiter=',', names=True)


        for k in ['train_loss', 'val_loss']:
            ax_1.plot(stats['epoch'],
                    stats[k], label=experiment_name + ' ' + label[k])


        for k in ['train_acc', 'val_acc']:
            ax_2.plot(stats['epoch'],
                    stats[k], label=experiment_name + ' ' + label[k])


        for k in ['val_vid_acc_top1']:
            ax_3.plot(stats['epoch'],
                    stats[k], label=experiment_name + ' ' + label[k])


        for k in ['val_vid_acc_top5']:
            ax_4.plot(stats['epoch'],
                    stats[k], label=experiment_name + ' ' + label[k])

    ax_1.legend(loc=0)
    ax_2.legend(loc=0)
    ax_3.legend(loc=0)
    ax_4.legend(loc=0)

    fig_1.tight_layout()
    fig_1.savefig(os.path.join(output_fig_dir, 'loss.pdf'))
    plt.close(fig_1)

    fig_2.tight_layout()
    fig_2.savefig(os.path.join(output_fig_dir, 'accuracy.pdf'))
    plt.close(fig_2)

    fig_3.tight_layout()
    fig_3.savefig(os.path.join(output_fig_dir, 'video_accuracy_top1.pdf'))
    plt.close(fig_3)

    fig_4.tight_layout()
    fig_4.savefig(os.path.join(output_fig_dir, 'video_accuracy_top5.pdf'))
    plt.close(fig_4)


if __name__ == '__main__':
    #plot_multiple_stats('/disk/scratch1/s1884147/model', ['RN_test2', 'RN_no_coordinates'])
    plot_multiple_stats(args.experiment_root, args.dataset, args.experiment_names)
