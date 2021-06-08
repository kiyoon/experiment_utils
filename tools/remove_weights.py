import os
import shutil

from experiment_utils import ExperimentBuilder


def remove_exp_dir(exp):
    # empty experiment
    remove = input("This directory appears to be empty. Remove the entire directory? (y/n) ")
    if remove.lower() == 'y':
        try:
            shutil.rmtree(exp.experiment_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


def remove_weights(experiment_root, dataset, model, experiment_name):
    try:
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_singlelabel(multicropval=True)
        exp = ExperimentBuilder(experiment_root, dataset, model, experiment_name, summary_fieldnames, summary_fieldtypes)
        exp.load_summary()
        best_metric = 'val_acc'
    except KeyError:
        summary_fieldnames, summary_fieldtypes = ExperimentBuilder.return_fields_multilabel(multicropval=True)
        exp = ExperimentBuilder(experiment_root, dataset, model, experiment_name, summary_fieldnames, summary_fieldtypes)
        exp.load_summary()
        best_metric = 'val_vid_mAP'
    except FileNotFoundError:
        remove_exp_dir(exp)
        return

    if len(exp.summary['epoch']) == 0:
        remove_exp_dir(exp)
        return


    best_epoch = exp.get_best_model_stat(best_metric)['epoch']
    last_epoch = max(exp.summary['epoch'])
    remove_list = []
    nonexist_list = []
    keep_list = []

    # Search
    for epoch in exp.summary['epoch']:
        checkpoint_path = exp.get_checkpoint_path(epoch)
        if epoch not in [best_epoch, last_epoch]:
            if os.path.isfile(checkpoint_path):
                remove_list.append(checkpoint_path)
            else:
                nonexist_list.append(checkpoint_path)
        else:
            keep_list.append((checkpoint_path, exp.get_epoch_stat(epoch)))

    if len(nonexist_list) > 0:
        print("* We can't find these files:")
        for checkpoint_path in nonexist_list:
            print(checkpoint_path)
        print()

    print("* These files will be removed:")
    for checkpoint_path in remove_list:
        print(checkpoint_path)

    print()
    print("* These files will be kept:")
    for checkpoint_path, stat in keep_list:
        print(checkpoint_path, stat)
    print()
    print()

    if len(remove_list) > 0:
        proceed = input("Proceed? (y/n) ")
        if proceed.lower() == 'y':
            for remove_path in remove_list:
                os.remove(remove_path)

        print("Successfully removed.")
    else:
        print("Nothing to remove. Terminating..")
    
if __name__ == '__main__':
    from experiment_utils.argparse_utils import add_exp_arguments

    import argparse
    def get_parser():
        parser = argparse.ArgumentParser(description="Remove unnecessary weights. Remove everything except the best model and the last model.",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_exp_arguments(parser, dataset_choices=None)

        return parser

    parser = get_parser()
    args = parser.parse_args()


    remove_weights(args.experiment_root, args.dataset, args.model, args.experiment_name)
