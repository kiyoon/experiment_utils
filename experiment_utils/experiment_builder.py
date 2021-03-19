import os
import csv, json
from .csv_to_dict import csv_to_dict
from .plot_stats import plot_stats_singlelabel, plot_stats_multilabel
from .  import telegram_post as tg
import numpy as np
import matplotlib.pyplot as plt
import configparser
from .human_time_duration import human_time_duration

checkpoints_format = "epoch_{:04d}.pth"

"""Vocabularies I used
stat = 1 line of the summary
summary = 1 training experiment's stats
stats = could be the same as summary
summaries = multiple experiments' summaries
"""

class ExperimentBuilder():
    @staticmethod
    def return_fields_singlelabel(multicropval = True):
        summary_fieldnames = ['epoch', 'train_runtime_sec', 'train_loss', 'train_acc', 'val_runtime_sec', 'val_loss', 'val_acc']

        summary_fieldtypes = {'epoch': int, 'train_runtime_sec': float, 'train_loss': float, 'train_acc': float, 'val_runtime_sec': float, 'val_loss': float, 'val_acc': float}

        if multicropval:
            summary_fieldnames.extend(['multi_crop_val_runtime_sec', 'multi_crop_val_loss', 'multi_crop_val_acc', 'multi_crop_val_vid_acc_top1', 'multi_crop_val_vid_acc_top5'])
            summary_fieldtypes.update({'multi_crop_val_runtime_sec': float, 'multi_crop_val_loss': float, 'multi_crop_val_acc': float, 'multi_crop_val_vid_acc_top1': float, 'multi_crop_val_vid_acc_top5': float})

        return summary_fieldnames, summary_fieldtypes


    @staticmethod
    def return_fields_multilabel(multicropval = True):
        summary_fieldnames = ['epoch', 'train_runtime_sec', 'train_loss', 'train_vid_mAP', 'val_runtime_sec', 'val_loss', 'val_vid_mAP']

        summary_fieldtypes = {'epoch': int, 'train_runtime_sec': float, 'train_loss': float, 'train_vid_mAP': float, 'val_runtime_sec': float, 'val_loss': float, 'val_vid_mAP': float}

        if multicropval:
            summary_fieldnames.extend(['multi_crop_val_runtime_sec', 'multi_crop_val_loss', 'multi_crop_val_vid_mAP'])
            summary_fieldtypes.update({'multi_crop_val_runtime_sec': float, 'multi_crop_val_loss': float, 'multi_crop_val_vid_mAP': float})

        return summary_fieldnames, summary_fieldtypes

    def __init__(self, experiment_root, dataset, model_name, experiment_name, summary_fieldnames = None, summary_fieldtypes = None, telegram_key_ini = None):
        """Initialise the experiment common paths.

        Params:
            experiment_root (str): Root directory
            dataset (str): Name of the dataset
            model_name (str): Name of the model 
            experiment_name (str): Name of the experiment
        """
        self.experiment_root = experiment_root
        self.dataset = dataset
        self.model_name = model_name
        self.experiment_name = experiment_name

        # dirs
        self.experiment_dir = os.path.join(experiment_root, dataset, model_name, experiment_name)

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
            self.summary_fieldnames = ['epoch', 'train_runtime_sec', 'train_loss', 'train_acc', 'val_runtime_sec', 'val_loss', 'val_acc', 'multi_crop_val_runtime_sec', 'multi_crop_val_loss', 'multi_crop_val_acc', 'multi_crop_val_vid_acc_top1', 'multi_crop_val_vid_acc_top5']

        if summary_fieldtypes:
            self.summary_fieldtypes = summary_fieldtypes
        else:
            self.summary_fieldtypes = {'epoch': int, 'train_runtime_sec': float, 'train_loss': float, 'train_acc': float, 'val_runtime_sec': float, 'val_loss': float, 'val_acc': float, 'multi_crop_val_runtime_sec': float, 'multi_crop_val_loss': float, 'multi_crop_val_acc': float, 'multi_crop_val_vid_acc_top1': float, 'multi_crop_val_vid_acc_top5': float}


        if telegram_key_ini:
            key = configparser.ConfigParser()
            key.read(telegram_key_ini)
            self.tg_token = key['Telegram']['token']
            self.tg_chat_id = key['Telegram']['chat_id']
        else:
            self.tg_token = None
            self.tg_chat_id = None


    def make_dirs_for_training(self):
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)


    def get_checkpoint_path(self, epoch):
        return os.path.join(self.weights_dir, checkpoints_format.format(epoch))


    def get_epoch_stat(self, epoch):
        epoch_indices = [i for i, e in enumerate(self.summary['epoch']) if e == epoch]
        if len(epoch_indices) != 1:
            raise ValueError("Too many or no epoch found in the summary. Found %d epoch stats." % len(epoch_indices))
        
        epoch_idx = epoch_indices[0]

        stat = {}
        for fieldname in self.summary_fieldnames:
            stat[fieldname] = self.summary[fieldname][epoch_idx]

        return stat

    def get_best_model_stat(self, field = 'val_acc'):
        array_to_argmax = np.array([v if v != None else -100 for v in self.summary[field] ])
        best_idx = array_to_argmax.argmax()

        best_stat = {}
        for fieldname in self.summary_fieldnames:
            best_stat[fieldname] = self.summary[fieldname][best_idx]

        return best_stat


    def get_avg_value(self, field = 'train_runtime_sec'):
        values = np.array([v for v in self.summary[field] if v is not None])
        if values.size == 0:
            return 0

        average_value = values.mean()
        return average_value

    def get_sum_value(self, field = 'train_runtime_sec'):
        values = np.array([v for v in self.summary[field] if v is not None])
        if values.size == 0:
            return 0

        sum_value = values.sum()
        return sum_value 



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
            if fieldname in curr_stat.keys():
                self.summary[fieldname].append(curr_stat[fieldname])
            else:
                self.summary[fieldname].append(None)

        with open(self.summary_file, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.summary_fieldnames)
            csv_writer.writerow(curr_stat)


    def time_summary_to_text(self, perform_multicropval):
        train_time_avg = human_time_duration(self.get_avg_value('train_runtime_sec'))
        val_time_avg = human_time_duration(self.get_avg_value('val_runtime_sec'))
        train_time_sum = self.get_sum_value('train_runtime_sec')
        val_time_sum = self.get_sum_value('val_runtime_sec')
        if not perform_multicropval:
            text = f"Train/val time per epoch: {train_time_avg}, {val_time_avg}"
            text += f"\nTotal time elapsed: {human_time_duration(train_time_sum+val_time_sum)}"
        else:
            multicropval_time_avg = human_time_duration(self.get_avg_value('multi_crop_val_runtime_sec'))
            multicropval_time_sum = self.get_sum_value('multi_crop_val_runtime_sec')
            text = f"Train/val/multicropval time per epoch: {train_time_avg}, {val_time_avg}, {multicropval_time_avg}"
            text += f"\nTotal time elapsed: {human_time_duration(train_time_sum+val_time_sum+multicropval_time_sum)}"

        return text 

    def plot_summary(self, send_telegram = False):
        """Save summary plots to the plot dir and also send to Telegram
        """

        if 'train_acc' in self.summary_fieldnames:
            task = 'singlelabel_classification'
        elif 'train_vid_mAP' in self.summary_fieldnames:
            task = 'multilabel_classification'
        else:
            raise NotImplementedError("Unknown task, cannot plot stats")

        perform_multicropval = 'multi_crop_val_runtime_sec' in self.summary_fieldnames


        if task == 'singlelabel_classification':
            loss_fig, acc_fig, acc5_fig = plot_stats_singlelabel(self.summary, self.plots_dir)

            if send_telegram:
                best_clip_val_acc = self.get_best_model_stat('val_acc')
                text = "Plots at epoch {:d}\nHighest (at epoch {:d}) / Last clip val acc {:.4f} / {:.4f}".format(self.summary['epoch'][-1],
                    best_clip_val_acc['epoch'], best_clip_val_acc['val_acc'], self.summary['val_acc'][-1])

                if perform_multicropval:
                    best_video_val_acc = self.get_best_model_stat('multi_crop_val_vid_acc_top1')
                    text += "\nHighest (at epoch {:d}) / Last video val acc {:.4f} / {:.4f}".format(
                            best_video_val_acc['epoch'], best_video_val_acc['multi_crop_val_vid_acc_top1'], self.summary['multi_crop_val_vid_acc_top1'][-1])

                text += "\n" + self.time_summary_to_text(perform_multicropval)


                self.tg_send_text_with_expname(text)
                self.tg_send_matplotlib_fig(loss_fig)
                self.tg_send_matplotlib_fig(acc_fig)
                if acc5_fig:
                    self.tg_send_matplotlib_fig(acc5_fig)

            plt.close(loss_fig)
            plt.close(acc_fig)
            if acc5_fig:
                plt.close(acc5_fig)

        elif task == 'multilabel_classification':
            loss_fig, mAP_fig = plot_stats_multilabel(self.summary, self.plots_dir)
            if send_telegram:
                best_stat = self.get_best_model_stat('val_vid_mAP')
                text = "Plots at epoch {:d}\nHighest (at epoch {:d}) / Last video val mAP {:.4f} / {:.4f}".format(self.summary['epoch'][-1],
                    best_stat['epoch'], best_stat['val_vid_mAP'], self.summary['val_vid_mAP'][-1])

                if perform_multicropval:
                    best_multicrop_stat = self.get_best_model_stat('multi_crop_val_vid_mAP')
                    text += "\nHighest (at epoch {:d}) / Last multicrop video val mAP {:.4f} / {:.4f}".format(
                            best_multicrop_stat['epoch'], best_multicrop_stat['multi_crop_val_vid_mAP'], self.summary['multi_crop_val_vid_mAP'][-1])

                text += "\n" + self.time_summary_to_text(perform_multicropval)

                self.tg_send_text_with_expname(text)
                self.tg_send_matplotlib_fig(loss_fig)
                self.tg_send_matplotlib_fig(mAP_fig)

            plt.close(loss_fig)
            plt.close(mAP_fig)


    # Send telegram messages when telegram key is initialised.
    def tg_send_text(self, text, parse_mode = None):
        if self.tg_token:
            return tg.send_text(self.tg_token, self.tg_chat_id, text, parse_mode)
        return None

    def tg_send_text_with_title(self, title, body):
        if self.tg_token:
            return tg.send_text_with_title(self.tg_token, self.tg_chat_id, title, body)
        return None

    def tg_send_text_with_expname(self, body):
        return self.tg_send_text_with_title('{} {} {}'.format(self.dataset, self.model_name, self.experiment_name), body)

    def tg_send_photo(self, img_path):
        if self.tg_token:
            return tg.send_photo(self.tg_token, self.tg_chat_id, img_path)
        return None

    def tg_send_remote_photo(self, img_url):
        if self.tg_token:
            return tg.send_remote_photo(self.tg_token, self.tg_chat_id, img_url)
        return None

    def tg_send_matplotlib_fig(self, fig):
        if self.tg_token:
            return tg.send_matplotlib_fig(self.tg_token, self.tg_chat_id, fig)
        return None
