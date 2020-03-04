def add_exp_arguments(parser, dataset_choices=None, root_default=None, dataset_default=None, name_default=None):
    parser.add_argument("-r", "--experiment_root", type=str, required=root_default is None, default=root_default, help="A directory to save the experiment.")
    parser.add_argument("-d", "--dataset", type=str, required=dataset_default is None, default=dataset_default, choices=dataset_choices, help="Dataset to use.")
    parser.add_argument("-n", "--experiment_name", type=str, required=name_default is None, default=name_default, help="A directory name in experiment_root/dataset to save/access the model and logs.")
