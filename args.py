import argparse


def get_args():
    parser = argparse.ArgumentParser(description='CSP Metrics Arguments')

    parser.add_argument('--gt_path', type=str, default=None)
    parser.add_argument('--pred_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--method_name', type=str, default=None)
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--use_alignment', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_lcm', action='store_true', default=False)
    parser.add_argument('--max_lcm', type=int, default=0)

    return parser.parse_args()
