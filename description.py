import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer


def descriptions():
    # dataset_name = "BreastCancerWisconsin"
    dataset_name = "Iris"

    # Set dataset
    if dataset_name == "Iris":
        dataset = load_iris()
        original_labels = ['setosa', 'versicolour', 'virginica']
        n_features = 4
        n_columns = 2
        n_figures_per_column = 2
    elif dataset_name == "BreastCancerWisconsin":
        dataset = load_breast_cancer()
        original_labels = ['malignant', 'benign']
        n_features = 30
        n_columns = 3
        n_figures_per_column = 10
    else:
        raise ("Dataset not found")

    # Plotting
    fig, axes = plt.subplots(n_figures_per_column, n_columns, figsize=(n_figures_per_column + 2, n_columns * 3))

    labels = []
    for i in range(0, len(original_labels)):
        labels.append(dataset.data[dataset.target == i])

    colors = ['r', 'g', 'b']

    ax = axes.ravel()
    for i in range(n_features):
        _, bins = np.histogram(dataset.data[:, i], bins=40)

        for j in range(0, len(original_labels)):
            ax[i].hist(labels[j][:, i], bins=bins, color=colors[j], alpha=(1 / (2 + j)))

        ax[i].set_title(dataset.feature_names[i], fontsize=9)
        ax[i].axes.get_xaxis().set_visible(False)
        ax[i].set_yticks(())

    plt.tight_layout()
    ax[0].legend(original_labels, loc='best', fontsize=9)
    plt.savefig('{}.jpg'.format(dataset_name), format="svg")


if __name__ == "__main__":
    descriptions()
