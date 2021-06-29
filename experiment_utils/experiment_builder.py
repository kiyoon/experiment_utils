import os
import csv, json
from .plot_stats import plot_stats_singlelabel, plot_stats_multilabel
from .  import telegram_post as tg
import numpy as np
import matplotlib.pyplot as plt
import configparser
from .human_time_duration import human_time_duration

import logging
logger = logging.getLogger(__name__)

import pandas as pd

"""Vocabularies I used
stat = 1 line of the summary
summary = 1 training experiment's stats
stats = could be the same as summary
summaries = multiple experiments' summaries
"""

def DataFrame_loc_dict(df, index):
    """Similar to df.iloc[index], but that will spoil the data type.
    This function will return as Python dict which will preserve the type.
    """
    row_dict = {}
    for column in df.columns:
        row_dict[column] = df.at[index, column]
    return row_dict


class ExperimentBuilder():
    @staticmethod
    def return_fields_from_metrics(metrics):
        summary_fieldnames = ['epoch']
        summary_fieldtypes = {'epoch': int}

        for split in ['train', 'val', 'multicropval']:
            # Every split has runtime and loss metric by default.
            summary_fieldnames.extend([f'{split}_runtime_sec', f'{split}_loss'])
            summary_fieldtypes.update({f'{split}_runtime_sec': float, f'{split}_loss': float})

            if split in metrics.keys() and metrics[split] is not None and len(metrics[split]) > 0:
                for metric in metrics[split]:
                    fieldtypes = metric.types_of_metrics()
                    fieldnames = metric.get_csv_fieldnames(split)

                    if isinstance(fieldnames, tuple) and isinstance(fieldtypes, tuple):
                        for fieldname, fieldtype in zip(fieldnames, fieldtypes):
                            summary_fieldnames.append(fieldname)
                            summary_fieldtypes.update({fieldname: fieldtype})
                    elif isinstance(fieldnames, str) and isinstance(fieldtypes, type):
                        summary_fieldnames.append(fieldnames)
                        summary_fieldtypes.update({fieldnames: fieldtypes})
                    else:
                        raise ValueError('Metric definition error: types_of_metrics() and get_csv_fieldnames() has to return both tuple, or type and str.')

        return summary_fieldnames, summary_fieldtypes

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

    def __init__(self, experiment_root, dataset, model_name, experiment_name, summary_fieldnames = None, summary_fieldtypes = None, telegram_key_ini = None, telegram_bot_idx = 0, checkpoints_format = "epoch_{:04d}.pth"):
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

        # summary (pandas.DataFrame) and summary.csv should always be synchronised.
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
            try:
                self.tg_token = key[f'Telegram{telegram_bot_idx}']['token']
                self.tg_chat_id = key[f'Telegram{telegram_bot_idx}']['chat_id']
                if not self.tg_token or not self.tg_chat_id:
                    raise KeyError('No telegram bot given')
                else:
                    logger.info(f'Telegram bot initialised with keys in {telegram_key_ini} and using the bot number {telegram_bot_idx}')
            except KeyError as e:
                logger.warning(f'Telegram token and chat_id not found in {telegram_key_ini}. Suppressing all the Telegram outputs.')
                self.tg_token = None
                self.tg_chat_id = None
        else:
            logger.info(f'Telegram bot not initialised.')
            self.tg_token = None
            self.tg_chat_id = None

        self.checkpoints_format = checkpoints_format

    def make_dirs_for_training(self):
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)


    def get_checkpoint_path(self, epoch):
        return os.path.join(self.weights_dir, self.checkpoints_format.format(epoch))


    def get_epoch_stat(self, epoch):
        epoch_stat = self.summary[self.summary['epoch'] == epoch]
        if len(epoch_stat) != 1:
            raise ValueError("Too many or no epoch found in the summary. Found %d epoch stats." % len(epoch_stat))
        
        return epoch_stat.to_dict('records')[0]


    def get_best_model_stat(self, field = 'val_acc', is_better_func=lambda a,b: a>b):
        field_values = self.summary[field].dropna()
        if len(field_values) == 0:
            raise ValueError(f'No values in the field {field}')

        best_idx = field_values.index[0]
        best_value = field_values[best_idx]
        if len(field_values) == 1:
            # Using iloc will spoil the data type!
            return DataFrame_loc_dict(self.summary, best_idx)

        for idx in field_values.index[1:]:
            if is_better_func(field_values[idx], best_value):
                best_idx = idx
                best_value = field_values[idx]

        return DataFrame_loc_dict(self.summary, best_idx)


    def get_last_model_stat(self, field = 'val_acc'):
        """Ignore None values and get the last stat
        """
        field_values = self.summary[field].dropna()
        if len(field_values) == 0:
            raise ValueError(f'No values in the field {field}')

        last_idx = field_values.index[-1]
        return DataFrame_loc_dict(self.summary, last_idx)


    def get_avg_value(self, field = 'train_runtime_sec'):
        if self.summary[field].size == 0:
            return 0

        return self.summary[field].mean(skipna=True)

    def get_sum_value(self, field = 'train_runtime_sec'):
        if self.summary[field].size == 0:
            return 0

        return self.summary[field].sum(skipna=True)



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

        dataframe_columns = {}
        for fieldname in self.summary_fieldnames:
            dataframe_columns[fieldname] = pd.Series([], dtype=self.summary_fieldtypes[fieldname])

        self.summary = pd.DataFrame(dataframe_columns)


    def load_summary(self):
        self.summary = pd.read_csv(self.summary_file, dtype=self.summary_fieldtypes)
        fieldnames = list(self.summary.columns)
        if fieldnames != self.summary_fieldnames:
            logger.warning(f"self.summary_filednames does not match to that of summary.csv file. Ignoring the CSV fields {fieldnames}.")


    def add_summary_line(self, curr_stat):
        """Writes one line of training stat to self.summary_file and self.summary
        """
        self.summary.append(curr_stat)

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
                    last_video_val_acc = self.get_last_model_stat('multi_crop_val_vid_acc_top1')
                    text += "\nHighest (at epoch {:d}) / Last (at epoch {:d}) video val acc {:.4f} / {:.4f}".format(
                            best_video_val_acc['epoch'], last_video_val_acc['epoch'], best_video_val_acc['multi_crop_val_vid_acc_top1'], last_video_val_acc['multi_crop_val_vid_acc_top1'])

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
                    last_multicrop_stat = self.get_last_model_stat('multi_crop_val_vid_mAP')
                    text += "\nHighest (at epoch {:d}) / Last (at epoch {:d}) multicrop video val mAP {:.4f} / {:.4f}".format(
                            best_multicrop_stat['epoch'], last_multicrop_stat['epoch'], best_multicrop_stat['multi_crop_val_vid_mAP'], last_multicrop_stat['multi_crop_val_vid_mAP'])

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
