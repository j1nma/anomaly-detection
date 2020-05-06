import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def log(logfile, s):
    """ Log a string into a file and print it. """
    with open(logfile, 'a', encoding='utf8') as f:
        f.write(str(s))
        f.write("\n")
    print(s)


def get_args_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument(
        "-d",
        "--dataset",
        default="Iris",
        help="Name of the dataset to use: Iris, BreastCancer."
    )
    parser.add_argument(
        "-t",
        "--technique",
        default="PCA",
        help="Name of the dimensionality reduction technique: PCA, tSNE, MDS."
    )
    parser.add_argument(
        "-c",
        "--components",
        default=2,
        help="Number of components to consider for PCA, tSNE: 2, 3."
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=1910299034,
        help="Random seed."
    )
    parser.add_argument(
        "-od",
        "--outdir",
        default='results/'
    )

    return parser


def experiments(config_file):
    args = get_args_parser().parse_args(['@' + config_file])

    # Set seed
    np.random.seed(int(args.seed))

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + timestamp + '/'

    # Create results directory
    outdir_path = Path(outdir)
    if not outdir_path.is_dir():
        os.makedirs(outdir)

    # Logging
    logfile = outdir + 'log.txt'

    log(logfile, "Directory " + outdir + " created.")

    # Set dataset
    if args.dataset == 'Iris':
        dataset = load_iris()
        dataset_name = "Iris"
        original_labels = ['setosa', 'versicolour', 'virginica']
    elif args.dataset == 'BreastCancer':
        dataset = load_breast_cancer()
        dataset_name = "Breast Cancer Wisconsin"
        original_labels = ['malignant', 'benign']
    else:
        raise ("Dataset not found")

    data = dataset.data
    labels = dataset.target

    # Add labels and feature names
    df = pd.DataFrame(data, columns=dataset.feature_names)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))
    for i in range(0, len(original_labels)):
        df['label'].replace(str(i), original_labels[i], inplace=True)

    # Dataset analysis
    log(logfile, 'Size of the data: {} and labels: {}'.format(data.shape, labels.shape))
    log(logfile, 'Size of the reshaped dataframe: {}'.format(df.shape))
    log(logfile, df.head())
    log(logfile, df.tail())

    # Normalization of features
    data = df.loc[:, dataset.feature_names].values
    data = StandardScaler().fit_transform(data)

    # Set number of components
    n_components = int(args.components)

    # Set technique
    if args.technique == 'PCA':
        pca = PCA(n_components=n_components)
        data_transformed = pca.fit_transform(data)
        transformed_df = pd.DataFrame(data=data_transformed, columns=['PC ' + str(i + 1) for i in range(n_components)])
        log(logfile, 'Cumulative explained variation for {} principal components: {}'.format(n_components,
                                                                                             np.sum(
                                                                                                 pca.explained_variance_ratio_).round(
                                                                                                 decimals=3)))
    elif args.technique == 'tSNE':
        time_start = time.time()
        tsne = TSNE(n_components=n_components, n_iter=1000, random_state=int(args.seed))
        data_transformed = tsne.fit_transform(data)
        log(logfile, 't-SNE done! Time elapsed: {0:.3f} seconds'.format(time.time() - time_start))
        transformed_df = pd.DataFrame(data=data_transformed, columns=['PC ' + str(i + 1) for i in range(n_components)])

    elif args.technique == 'MDS':
        embedding = MDS(n_components=n_components)
        data_transformed = embedding.fit_transform(data)
        log(logfile, 'MDS transformation shape: {}'.format(data_transformed.shape))
        transformed_df = pd.DataFrame(data=data_transformed, columns=['PC ' + str(i + 1) for i in range(n_components)])

    else:
        raise ("Technique not found")

    log(logfile, transformed_df.tail())

    # Plotting
    if n_components == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        ax.set_zlabel('PC 3', fontsize=15)
    elif n_components == 2:
        plt.figure(figsize=(10, 10))

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('PC 1', fontsize=15)
    plt.ylabel('PC 2', fontsize=15)

    plt.title("{} of {} Dataset".format(args.technique, dataset_name), fontsize=20)
    colors = ['r', 'g', 'b']
    for label, color in zip(original_labels, colors):
        indicesToKeep = df['label'] == label
        if n_components == 3:
            ax.scatter(transformed_df.loc[indicesToKeep, 'PC 1'],
                       transformed_df.loc[indicesToKeep, 'PC 2'],
                       transformed_df.loc[indicesToKeep, 'PC 3'], c=color, s=50)
        else:
            plt.scatter(transformed_df.loc[indicesToKeep, 'PC 1'],
                        transformed_df.loc[indicesToKeep, 'PC 2'], c=color, s=50)

    if args.dataset == 'Iris':
        plt.legend(original_labels, prop={'size': 15}, loc="lower right")
    else:
        plt.legend(original_labels, prop={'size': 15}, loc="upper right")

    plt.savefig(outdir + '{}_c={}.svg'.format(args.technique, n_components), format="svg")


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
