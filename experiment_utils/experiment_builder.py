import os
import csv, json
from .  import telegram_post as tg
import numpy as np
import matplotlib.pyplot as plt
import configparser
from .human_time_duration import human_time_duration

import logging
logger = logging.getLogger(__name__)

import pandas as pd
import re
import shutil

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
    def return_fields_from_metrics(metrics: dict):
        summary_fieldnames = ['epoch']
        summary_fieldtypes = {'epoch': int}

        for split in ['train', 'val', 'multicropval']:
            # Every split has runtime and loss metric by default.
            summary_fieldnames.extend([f'{split}_runtime_sec', f'{split}_loss'])
            summary_fieldtypes.update({f'{split}_runtime_sec': float, f'{split}_loss': float})

            if split in metrics.keys() and metrics[split] is not None and len(metrics[split]) > 0:
                for metric in metrics[split]:
                    fieldtypes = metric.types_of_metrics()
                    fieldnames = metric.get_csv_fieldnames()

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

    def __init__(self, experiment_root, dataset, model_name, experiment_name, summary_fieldnames, summary_fieldtypes, version = -2, telegram_key_ini = None, telegram_bot_idx = 0,
            checkpoints_format = "epoch_{:04d}.pth",
            version_format = "version_{:03d}", version_regex = "version_(\d+)", version_regex_group = 1):
        """Initialise the experiment common paths.

        Params:
            experiment_root (str): Root directory
            dataset (str): Name of the dataset
            model_name (str): Name of the model 
            experiment_name (str): Name of the experiment
            version (int): -2 last version for accessing existing experiment, -1 new version for new training, 0, 1, 2, ...
        """
        self.experiment_root = experiment_root
        self.dataset = dataset
        self.model_name = model_name
        self.experiment_name = experiment_name

        # version
        if version >= 0:
            self.experiment_dir = os.path.join(experiment_root, dataset, model_name, experiment_name, version_format.format(version))
            self.version = version
        else:
            # check if using old structure without version
            if os.path.isdir(os.path.join(experiment_root, dataset, model_name, experiment_name, 'logs')):
                # Treat it as version 0
                highest_version = 0
            else:
                # list existing versions and automatically assign version
                experiment_dir = os.path.join(experiment_root, dataset, model_name, experiment_name)
                if not os.path.isdir(experiment_dir):
                    highest_version = -1 #self.version = 0
                else:
                    highest_version = -1
                    dirnames = next(os.walk(experiment_dir))[1]
                    for dirname in dirnames:
                        regex_search = re.search(version_regex, dirname)
                        if regex_search is not None:
                            searched_version = int(regex_search.group(version_regex_group))
                            if searched_version > highest_version:
                                highest_version = searched_version

            if version == -1:
                self.version = highest_version + 1
            elif version == -2:
                self.version = highest_version
                assert self.version >= 0, 'No experiment version can be found. Does the experiment exist?'
            else:
                raise ValueError(f'version has to be one of -2 (last version), -1 (new version), or 0, 1, 2, ...., but got {version}')

            self.experiment_dir = os.path.join(experiment_root, dataset, model_name, experiment_name, version_format.format(self.version))

        # dirs
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


        self.summary_fieldnames = summary_fieldnames
        self.summary_fieldtypes = summary_fieldtypes


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
        self.version_format = version_format

    def _check_old_structure_and_move(self):
        old_experiment_dir = os.path.abspath(os.path.join(self.experiment_dir, os.pardir))
        if os.path.isdir(os.path.join(old_experiment_dir, 'logs')):
            # Old structure without version exists. Move this to version 0
            version_dir = self.version_format.format(0)
            logger.info(f'{old_experiment_dir} is using old structure without version. Moving the files to {version_dir}.')
            old_exp_dir_version = os.path.join(old_experiment_dir, version_dir)
            os.makedirs(old_exp_dir_version)    # exist not okay
            shutil.move(os.path.join(old_experiment_dir, 'configs'), old_exp_dir_version)
            shutil.move(os.path.join(old_experiment_dir, 'logs'), old_exp_dir_version)
            shutil.move(os.path.join(old_experiment_dir, 'plots'), old_exp_dir_version)
            shutil.move(os.path.join(old_experiment_dir, 'weights'), old_exp_dir_version)
            shutil.move(os.path.join(old_experiment_dir, 'tensorboard_runs'), old_exp_dir_version)
            if os.path.isdir(os.path.join(old_experiment_dir, 'predictions')):
                shutil.move(os.path.join(old_experiment_dir, 'predictions'), old_exp_dir_version)

    def make_dirs_for_training(self):
        self._check_old_structure_and_move()

        # Make dirs for current experiment
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.weights_dir, exist_ok=True)

    def copy_from(self, exp, copy_dirs=True, exclude_weights=True):
        """
        Instead of making dirs, initialise from another ExperimentBuilder.
        It doesn't copy the full state dict, so the directories and version etc. don't change and should stay different from the other exp.
        Instead, it copies the self.summary and copy the entire experiment folder.

        Raise exception when the directory already exists and you're trying to initialise (copy).
        """
        self.summary = exp.summary
        if copy_dirs:
            if exclude_weights:
                ignore_func = lambda directory, contents: contents if directory == exp.weights_dir else []
            else:
                ignore_func = None

            shutil.copytree(exp.experiment_dir, self.experiment_dir, symlinks=True, dirs_exist_ok=False,
                    ignore=ignore_func)

    def clip_summary_from_epoch(self, clip_from_epoch: int):
        """
        When you're loading from intermediate checkpoint (not last) and continuing experiment,
        you should remove later stats using this function, so none of the epoch stats get overlapped.
        """
        mask = self.summary.epoch < clip_from_epoch
        self.summary = self.summary[mask]

        # Update self.summary_file
        self.summary.to_csv(self.summary_file, index=False)


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
        if self.summary[field].count() == 0:
            return 0

        return self.summary[field].mean(skipna=True)

    def get_sum_value(self, field = 'train_runtime_sec'):
        if self.summary[field].count() == 0:
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
        self._check_old_structure_and_move()

        self.summary = pd.read_csv(self.summary_file, dtype=self.summary_fieldtypes)
        fieldnames = list(self.summary.columns)
        if fieldnames != self.summary_fieldnames:
            logger.warning(f"self.summary_filednames does not match to that of summary.csv file. Ignoring the CSV fields {fieldnames}.")


    def add_summary_line(self, curr_stat: dict):
        """Writes one line of training stat to self.summary_file and self.summary
        """
        self.summary = self.summary.append([curr_stat], ignore_index=True)  # appending list of dict is going to cast dict to DataFrame, which will ensure the dtypes are preserved. Otherwise, epoch will change to float type.

        with open(self.summary_file, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.summary_fieldnames)
            csv_writer.writerow(curr_stat)


    def update_summary_line(self, curr_stat: dict):
        """Update one line of training stat to self.summary_file and self.summary.
        It will find the row of curr_stat['epoch'], and update the other columns.
        """

        # update self.summary
        row_logicidx = self.summary.epoch == curr_stat['epoch']
        num_rows_found = row_logicidx.sum()
        if num_rows_found == 0:
            raise ValueError(f'No row found for the epoch {curr_stat["epoch"]}.')
        elif num_rows_found > 1:
            raise ValueError(f'More than 1 rows were found for the epoch {curr_stat["epoch"]}.')

        for fieldname, value in curr_stat.items():
            if fieldname not in self.summary_fieldnames:
                raise ValueError(f'You are trying to update summary field {fieldname} but it is not found for this experiment.')

            if fieldname == 'epoch':
                continue

            self.summary.loc[row_logicidx, fieldname] = value


        # Update self.summary_file
        self.summary.to_csv(self.summary_file, index=False)


    def time_summary_to_text(self):
        train_time_avg = human_time_duration(self.get_avg_value('train_runtime_sec'))
        val_time_avg = human_time_duration(self.get_avg_value('val_runtime_sec'))
        train_time_sum = self.get_sum_value('train_runtime_sec')
        val_time_sum = self.get_sum_value('val_runtime_sec')
        if 'multicropval_runtime_sec' not in self.summary.columns or self.summary['multicropval_runtime_sec'].count() == 0:
            text = f"Train/val time per epoch: {train_time_avg}, {val_time_avg}"
            text += f"\nTotal time elapsed: {human_time_duration(train_time_sum+val_time_sum)}"
        else:
            multicropval_time_avg = human_time_duration(self.get_avg_value('multicropval_runtime_sec'))
            multicropval_time_sum = self.get_sum_value('multicropval_runtime_sec')
            text = f"Train/val/multicropval time per epoch: {train_time_avg}, {val_time_avg}, {multicropval_time_avg}"
            text += f"\nTotal time elapsed: {human_time_duration(train_time_sum+val_time_sum+multicropval_time_sum)}"

        return text 


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
        return self.tg_send_text_with_title(f'{self.dataset} {self.model_name} {self.experiment_name} v{self.version}', body)

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
