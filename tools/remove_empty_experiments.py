import os
import shutil
import re

if __name__ == '__main__':
    import argparse
    def get_parser():
        parser = argparse.ArgumentParser(description="Remove unnecessary experiments, with no summary.csv file or empty file, with no weights. Assume every directory starting with `version_` are the experiment folders.",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('root_dir')
        parser.add_argument('--version_regex', default='version_(\d+)', help='Directory name pattern to search for.')
        parser.add_argument('--dry_run', action = 'store_true', help='Do not actually delete and see what it is going to delete.')

        return parser

    parser = get_parser()
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.root_dir, topdown=False):
        for dirname in dirs:
            regex_search = re.search(args.version_regex, dirname)
            if regex_search is not None:
                exp_dir = os.path.join(root, dirname)
                summary_file = os.path.join(exp_dir, 'logs', 'summary.csv')
                is_no_weights = len(os.listdir(os.path.join(exp_dir, 'weights'))) == 0
                is_summary_empty = False
                if not os.path.isfile(summary_file):
                    is_summary_empty = True
                else:
                    with open(summary_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) < 2:
                            # Only contains header
                            is_summary_empty = True

                if is_summary_empty and is_no_weights:
                    print(f'Removing empty experiment dir {exp_dir}')
                    if not args.dry_run:
                        shutil.rmtree(exp_dir)


