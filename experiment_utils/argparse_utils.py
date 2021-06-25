def add_exp_arguments(parser, dataset_choices=None, model_choices=None, root_default=None, dataset_default=None, model_default=None, name_default=None, dataset_channel_choices=None, model_channel_choices=None, exp_channel_choices=None, dataset_channel_default=None, model_channel_default=None, exp_channel_default=None):

    parser.add_argument("-R", "--experiment_root", type=str, required=root_default is None, default=root_default, help="A directory to save the experiment.")
    parser.add_argument("-D", "--dataset", type=str, required=dataset_default is None, default=dataset_default, choices=dataset_choices, help="Dataset to use.")
    parser.add_argument("-M", "--model", type=str, required=model_default is None, default=model_default, choices=model_choices, help="Model architecture to use.")
    parser.add_argument("-E", "--experiment_name", type=str, required=name_default is None, default=name_default, help="A directory name in experiment_root/dataset to save/access the model and logs.")
    parser.add_argument("-c:d", "--dataset_channel", type=str, default=dataset_channel_default, choices=dataset_channel_choices, help="Dataset config channel.")
    parser.add_argument("-c:m", "--model_channel", type=str, default=model_channel_default, choices=model_channel_choices, help="Model config channel.")
    parser.add_argument("-c:e", "--experiment_channel", type=str, default=exp_channel_default, choices=exp_channel_choices, help="Dataset config channel.")
