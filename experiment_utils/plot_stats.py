#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys, os


# !! This makes a huge numpy and pandas compatibility issues. 
#import warnings
#warnings.filterwarnings('error')

plt.rcParams.update({'font.size': 16})


def plot_stats(stats, output_fig_dir):
    label = {'train_loss': 'training loss', 'train_acc': 'training accuracy', 'val_loss': 'validation clip loss', 'val_acc': 'validation clip accuracy', 'val_vid_acc_top1': 'validation video accuracy top-1', 'val_vid_acc_top5': 'validation video accuracy top-5'}

    #print(stats['train_loss'])

    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
#    ax_1.set_xlim([0,100])
#    ax_1.set_ylim([0,1])
    ax_1.set_xlim(auto=True)
    ax_1.set_ylim(auto=True)
    for k in ['train_loss', 'val_loss']:
        ax_1.plot(stats['epoch'],
                stats[k], label=label[k])
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    fig_1.tight_layout()
    fig_1.savefig(os.path.join(output_fig_dir, 'loss.pdf'))
    plt.close(fig_1)

    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    ax_1.set_xlim(auto=True)
    ax_1.set_ylim(auto=True)
    for k in ['train_acc', 'val_acc']:
        ax_1.plot(stats['epoch'],
                stats[k], label=label[k])
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    fig_1.tight_layout()
    fig_1.savefig(os.path.join(output_fig_dir, 'accuracy.pdf'))
    plt.close(fig_1)

    if 'val_vid_acc_top1' in stats.keys():
        fig_1 = plt.figure(figsize=(8, 4))
        ax_1 = fig_1.add_subplot(111)
    #    ax_1.set_xlim([0,100])
    #    ax_1.set_ylim([0,1])
        ax_1.set_xlim(auto=True)
        ax_1.set_ylim(auto=True)
        for k in ['val_vid_acc_top1']:
            ax_1.plot(stats['epoch'],
                    stats[k], label=label[k])
        ax_1.legend(loc=0)
        ax_1.set_xlabel('Epoch number')

        fig_1.tight_layout()
        fig_1.savefig(os.path.join(output_fig_dir, 'video_accuracy_top1.pdf'))
        plt.close(fig_1)

    if 'val_vid_acc_top5' in stats.keys():
        fig_1 = plt.figure(figsize=(8, 4))
        ax_1 = fig_1.add_subplot(111)
        ax_1.set_xlim(auto=True)
        ax_1.set_ylim(auto=True)
        for k in ['val_vid_acc_top5']:
            ax_1.plot(stats['epoch'],
                    stats[k], label=label[k])
        ax_1.legend(loc=0)
        ax_1.set_xlabel('Epoch number')

        fig_1.tight_layout()
        fig_1.savefig(os.path.join(output_fig_dir, 'video_accuracy_top5.pdf'))
        plt.close(fig_1)


