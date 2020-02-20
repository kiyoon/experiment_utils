import os
import csv, json
from .csv_to_dict import csv_to_dict
from .plot_stats import plot_stats
import numpy as np

checkpoints_format = "epoch_{:04d}.pth"

"""Vocabularies I used
stat = 1 line of the summary
summary = 1 training experiment's stats
stats = could be the same as summary
summaries = multiple experiments' summaries
"""

class ExperimentBuilder():
    def __init__(self, experiment_root, dataset, experiment_name, summary_fieldnames = None, summary_fieldtypes = None):
        """Initialise the experiment common paths.

        Params:
            experiment_root (str): Root directory
            dataset (str): Name of the dataset
            experiment_name (str): Name of the experiment
        """
        self.experiment_root = experiment_root
        self.dataset = dataset
        self.experiment_name = experiment_name

        # dirs
        self.experiment_dir = os.path.join(experiment_root, dataset, experiment_name)

        self.configs_dir = os.path.join(self.experiment_dir, 'configs')
        self.logs_dir = os.path.join(self.experiment_dir, 'logs')
        self.plots_dir = os.path.join(self.experiment_dir, 'plots')
        self.weights_dir = os.path.join(self.experiment_dir, 'weights')

        # dirs (extra)
        self.tensorboard_runs_dir = os.path.join(self.experiment_dir, 'tensorboard_runs')
        self.predictions_dir = os.path.join(self.experiment_dir, 'predictions')

        # files
        self.args_file = os.path.join(self.configs_dir, 'args.json')
        self.summary_file = os.path.join(self.logs_dir, 'summary.csv')

        # summary (dict) and summary.csv should always be synchronised.
        self.summary = None


        if summary_fieldnames:
            self.summary_fieldnames = summary_fieldnames
        else:
            self.summary_fieldnames = ['epoch', 'runtime_sec', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_vid_acc_top1', 'val_vid_acc_top5']

        if summary_fieldtypes:
            self.summary_fieldtypes = summary_fieldtypes
        else:
            self.summary_fieldtypes = {'epoch': int, 'runtime_sec': float, 'train_loss': float, 'train_acc': float, 'val_loss': float, 'val_acc': float, 'val_vid_acc_top1': float, 'val_vid_acc_top5': float}


    def make_dirs_for_training(self):
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)


    def get_checkpoint_path(self, epoch):
        return os.path.join(self.weights_dir, checkpoints_format.format(epoch))


    def get_best_model_stat(self, field = 'val_vid_acc_top1'):
        array_to_argmax = np.array(self.summary[field])
        best_idx = array_to_argmax.argmax()

        best_stat = {}
        for fieldname in self.summary_fieldnames:
            best_stat[fieldname] = self.summary[fieldname][best_idx]

        return best_stat


    def dump_args(self, args, open_mode = 'a'):
        """Save args as a json file to record the config of the experiment.
        """
        with open(self.args_file, open_mode) as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=4)


    def load_args_json(self):
        with open(self.args_file, 'r') as f:
            exp_args = json.load(f)
        return exp_args


    def init_summary(self):
        """Return empty summary dictionary that will include training stats. Also write summary.csv header.
        """
        assert not os.path.isfile(self.summary_file), "Summary file '%s' already exists. Cannot initialise." % self.summary_file

        with open(self.summary_file, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.summary_fieldnames)
            csv_writer.writeheader()

        self.summary = {}
        for fieldname in self.summary_fieldnames:
            self.summary[fieldname] = []


    def load_summary(self):
        self.summary = csv_to_dict(self.summary_file, type_convert = self.summary_fieldtypes)


    def add_summary_line(self, curr_stat):
        """Writes one line of training stat to self.summary_file and self.summary
        """

        for fieldname in self.summary_fieldnames:
            self.summary[fieldname].append(curr_stat[fieldname])

        with open(self.summary_file, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.summary_fieldnames)
            csv_writer.writerow(curr_stat)


    def plot_summary(self):
        plot_stats(self.summary, self.plots_dir)




