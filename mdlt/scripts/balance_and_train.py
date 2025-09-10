import argparse
import os
import pandas as pd
import numpy as np
import subprocess


def kl_divergence(counts):
    probs = counts / counts.sum()
    uniform = np.ones_like(probs) / len(probs)
    return float((probs * np.log(probs / uniform + 1e-12)).sum())


def balance_dataset(csv_path, ratio=1.0, seed=0):
    df = pd.read_csv(csv_path)
    df_train = df[df['split'] == 'train']
    rng = np.random.RandomState(seed)

    counts = df_train['label'].value_counts().sort_index()
    min_count = counts.min()
    target = int(min_count * ratio)

    selected_idx = []
    for env in df_train['env'].unique():
        env_df = df_train[df_train['env'] == env]
        for label in counts.index:
            label_idx = env_df[env_df['label'] == label].index
            n = min(target, len(label_idx))
            if n > 0:
                selected_idx.extend(rng.choice(label_idx, n, replace=False))

    balanced_df = pd.concat([df.loc[selected_idx], df[df['split'] != 'train']])
    balanced_df.sort_index(inplace=True)
    return df, balanced_df


def main():
    parser = argparse.ArgumentParser(description="Create balanced subset and train")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='Fraction of the minority class to keep')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_steps', type=int, default=None)
    args = parser.parse_args()

    dset_dir = os.path.join(args.data_dir, args.dataset if args.dataset != 'DomainNet' else 'domain_net')
    csv_path = os.path.join(dset_dir, f"{args.dataset}.csv")
    orig_df, bal_df = balance_dataset(csv_path, args.ratio, args.seed)

    orig_counts = orig_df[orig_df['split'] == 'train']['label'].value_counts().sort_index()
    bal_counts = bal_df[bal_df['split'] == 'train']['label'].value_counts().sort_index()
    print('Original KL-divergence:', kl_divergence(orig_counts))
    print('Balanced KL-divergence:', kl_divergence(bal_counts))

    out_csv = os.path.join(dset_dir, f"{args.dataset}_balanced.csv")
    bal_df.to_csv(out_csv, index=False)

    env = os.environ.copy()
    env['MDLT_CSV_SUFFIX'] = '_balanced'
    cmd = [
        'python', 'mdlt/train.py',
        '--dataset', args.dataset,
        '--algorithm', 'ERM',
        '--output_folder_name', 'balanced',
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir
    ]
    if args.train_steps:
        cmd += ['--steps', str(args.train_steps)]

    subprocess.run(cmd, env=env, check=True)


if __name__ == '__main__':
    main()
