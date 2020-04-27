from experiment_utils.plot_stats import plot_stats
#from experiment_utils import ExperimentBuilder
from random import random

summary_fieldnames = ['epoch', 'train_runtime_sec', 'train_loss', 'train_acc', 'val_runtime_sec', 'val_loss', 'val_acc', 'multi_crop_val_runtime_sec', 'multi_crop_val_loss', 'multi_crop_val_acc', 'multi_crop_val_vid_acc_top1', 'multi_crop_val_vid_acc_top5']

def init_summary():
    """Return empty summary dictionary that will include training stats. Also write summary.csv header.
    """
    summary = {}

    for fieldname in summary_fieldnames:
        summary[fieldname] = []

    return summary

def add_summary_line(summary, curr_stat):
    """Writes one line of training stat to self.summary_file and self.summary
    """

    for fieldname in summary_fieldnames:
        if fieldname in curr_stat.keys():
            summary[fieldname].append(curr_stat[fieldname])
        else:
            summary[fieldname].append(None)


if __name__ == '__main__':
    elapsed_time = 10
    val_elapsed_time = 3
    multi_crop_val_elapsed_time = 100

    loss = 100.0
    acc = 0.01
    val_loss = 100.0
    val_acc = 0.01
    multi_crop_val_loss = 100.0
    multi_crop_val_acc = 0.01
    
    multi_crop_val_vid_acc_top1 = 0.10
    multi_crop_val_vid_acc_top5 = 0.30


    summary = init_summary()


    for epoch in range(100):
        curr_stat = {'epoch': epoch, 'train_runtime_sec': elapsed_time, 'train_loss': loss, 'train_acc': acc}

        curr_stat.update({'val_runtime_sec': val_elapsed_time, 'val_loss': val_loss, 'val_acc': val_acc})

        loss -= random()
        acc += random()
        val_loss -= random()
        val_acc += random()
        multi_crop_val_loss -= random()
        multi_crop_val_acc += random()

        if epoch % 5 == 4:
            curr_stat.update({'multi_crop_val_runtime_sec': multi_crop_val_elapsed_time, 'multi_crop_val_loss': multi_crop_val_loss, 'multi_crop_val_acc': multi_crop_val_acc, 'multi_crop_val_vid_acc_top1': multi_crop_val_vid_acc_top1, 'multi_crop_val_vid_acc_top5': multi_crop_val_vid_acc_top5})
            multi_crop_val_vid_acc_top1 += random() * 5
            multi_crop_val_vid_acc_top5 += random() * 5

        
        add_summary_line(summary, curr_stat)

    plot_stats(summary, 'test_plots')

