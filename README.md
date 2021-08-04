# Experiment Utils

It will make experiment directories for you, and enable easy access to commonly-used file paths (e.g. loading weights that has the highest validation accuracy). Also, writing and reading from configs and logs will be easy.

The experiment directory will look something like this:  

```bash
${experiment_root}
└── ${dataset}
    └── ${model_name}
        └── ${experiment_name}
            └── version_000
                ├── configs
                │   └── args.json
                ├── logs
                │   └── summary.csv
                ├── plots
                │   ├── accuracy.pdf
                │   ├── loss.pdf
                │   ├── video_accuracy_top1.pdf
                │   └── video_accuracy_top5.pdf
                ├── (predictions)
                ├── tensorboard_runs
                └── weights
                    ├── epoch_0000.pth
                    └── epoch_0001.pth
```

