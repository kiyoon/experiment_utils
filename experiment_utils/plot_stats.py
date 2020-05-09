#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')       # Default TKinter backend instantiates windows even when saving the plot to files, causing problems.
import numpy as np
import sys, os


# !! This makes a huge numpy and pandas compatibility issues. 
#import warnings
#warnings.filterwarnings('error')

plt.rcParams.update({'font.size': 16})


def plot_stats(stats, output_fig_dir):
    label = {'train_loss': 'training loss', 'train_acc': 'training accuracy', 'val_loss': 'validation clip loss', 'val_acc': 'validation clip accuracy', 'multi_crop_val_loss': 'multi-crop validation clip loss', 'multi_crop_val_acc': 'multi-crop validation clip acc', 'multi_crop_val_vid_acc_top1': 'multi-crop validation video accuracy top-1', 'multi_crop_val_vid_acc_top5': 'multi-crop validation video accuracy top-5'}

    #print(stats['train_loss'])

    loss_fig = plt.figure(figsize=(8, 4))
    ax_1 = loss_fig.add_subplot(111)
#    ax_1.set_xlim([0,100])
#    ax_1.set_ylim([0,1])
    ax_1.set_xlim(auto=True)
    ax_1.set_ylim(auto=True)
    for k in ['train_loss', 'val_loss']:
        ax_1.plot(stats['epoch'],
                stats[k], label=label[k])
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    loss_fig.tight_layout()
    loss_fig.savefig(os.path.join(output_fig_dir, 'loss.pdf'))



    acc_fig = plt.figure(figsize=(8, 4))
    ax_1 = acc_fig.add_subplot(111)
    ax_1.set_xlim(auto=True)
    ax_1.set_ylim(auto=True)
    for k in ['train_acc', 'val_acc']:
        ax_1.plot(stats['epoch'],
                stats[k], label=label[k])

    key = 'multi_crop_val_vid_acc_top1'
    if key in stats.keys():
        valid_indices = [i for i,v in enumerate(stats[key]) if v != None]
        if len(valid_indices) > 0:
            valid_epoch = [stats['epoch'][v] for i,v in enumerate(valid_indices)]
            valid_acc_values = [stats[key][v] for i,v in enumerate(valid_indices)]
            for k in [key]:
                ax_1.plot(valid_epoch,
                        valid_acc_values, label=label[k])

    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    acc_fig.tight_layout()
    acc_fig.savefig(os.path.join(output_fig_dir, 'accuracy.pdf'))




    key = 'multi_crop_val_vid_acc_top5'
    acc5_fig = None
    if key in stats.keys():
        valid_indices = [i for i,v in enumerate(stats[key]) if v != None]
        if len(valid_indices) > 0:
            valid_epoch = [stats['epoch'][v] for i,v in enumerate(valid_indices)]
            valid_acc_values = [stats[key][v] for i,v in enumerate(valid_indices)]
            acc5_fig = plt.figure(figsize=(8, 4))
            ax_1 = acc5_fig.add_subplot(111)
        #    ax_1.set_xlim([0,100])
        #    ax_1.set_ylim([0,1])
            ax_1.set_xlim(auto=True)
            ax_1.set_ylim(auto=True)
            for k in [key]:
                ax_1.plot(valid_epoch,
                        valid_acc_values, label=label[k])
            ax_1.legend(loc=0)
            ax_1.set_xlabel('Epoch number')

            acc5_fig.tight_layout()
            acc5_fig.savefig(os.path.join(output_fig_dir, 'accuracy_top5.pdf'))
    
    return loss_fig, acc_fig, acc5_fig
