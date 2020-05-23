import argparse
import datetime
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from scipy.stats import multivariate_normal
from sklearn import semi_supervised
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report, fbeta_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelSpreading

NOT_FRAUD = 0
FRAUD = 1
SUPERVISED = 2
SEMI_SUPERVISED = 3
UNSUPERVISED = 4


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
        default="creditcard",
        help="Name of the dataset to use: creditcard, ieee."
    )
    parser.add_argument(
        "-m",
        "--method",
        default="ocSVM",
        help="Name of the outlier detection method: ocSVM, LOF, twoClass."
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


from matplotlib.patches import Ellipse


####
# Function at https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
####
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


####
# Function adapted from https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
####
def plot_gmm(gmm, X, label=True, ax=None, outdir=None):
    ax = ax or plt.gca()
    ax.set_title("GMM Plot: V17, V14")

    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X['V17'], X['V14'], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X['V17'], X['V14'], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.5 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos[np.ix_([17, 14])], covar[np.ix_([17, 14], [17, 14])], alpha=w * w_factor)

    plt.savefig(outdir + 'gmm_v17_v14.png', bbox_inches='tight')
    plt.clf()

    return True


def experiments(config_file):
    # Save cwd path
    cwd = str(Path.cwd())

    # Parse arguments
    args = get_args_parser().parse_args(['@' + config_file])

    # Set seed
    np.random.seed(int(args.seed))

    # Construct output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir + str(args.dataset) + "/" + timestamp + '/'

    # Create results directory
    outdir_path = Path(outdir)
    if not outdir_path.is_dir():
        os.makedirs(outdir)

    # Logging
    logfile = outdir + 'log.txt'
    log(logfile, "Directory " + outdir + " created.")

    # Set dataset
    if str(args.dataset) == 'creditcard':
        dataset_name = 'Credit Card'
        df = pd.read_csv('{}/data/creditcard/creditcard.csv'.format(cwd))  # TODO explain README

        # TODO Describe: head() info() describe() etc

        # Normalize 'Amount' column
        df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

        # Normalize 'Time' column
        df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

        X = df.drop(['Class'], axis=1)
        y = df['Class']

        # Split dataframe by class
        df_NF = df[df.Class == NOT_FRAUD]
        df_F = df[df.Class == FRAUD]

        # Split by data and labels
        X_NF, X_F = df_NF.drop(['Class'], axis=1), df_F.drop(['Class'], axis=1)
        y_NF, y_F = df_NF['Class'], df_F['Class']

        ### SUPERVISED ###
        # Split 80-20
        X_train_F_NF, X_test_F_NF, y_train_F_NF, y_test_F_NF = train_test_split(X, y,
                                                                                test_size=0.2,
                                                                                random_state=int(args.seed))

        ### SEMI-SUPERVISED ###
        # Split 80-20 NF
        X_train_NF, X_cv_test_NF, y_train_NF, y_cv_test_NF = train_test_split(X_NF, y_NF,
                                                                              test_size=0.2,
                                                                              random_state=int(args.seed))

        # Split extra 20 NF for cross validation and testing
        X_cv_NF, X_test_NF, y_cv_NF, y_test_NF = train_test_split(X_cv_test_NF, y_cv_test_NF,
                                                                  test_size=0.5,
                                                                  random_state=int(args.seed))

        # Split 50-50 F data
        X_cv_F, X_test_F, y_cv_F, y_test_F = train_test_split(X_F, y_F, test_size=0.5, random_state=int(args.seed))

        # Build cross validation and testing sets
        X_cv = np.vstack([X_cv_NF, X_cv_F])
        y_cv = np.hstack([y_cv_NF, y_cv_F])
        X_test = np.vstack([X_test_NF, X_test_F])
        y_test = np.hstack([y_test_NF, y_test_F])

    elif str(args.dataset) == 'ieee':
        dataset_name = 'IEEE'
        train_transaction = pd.read_csv('{}/data/ieee/train_transaction.csv'.format(cwd), header=None,
                                        engine='python')  # TODO explain README
        train_identity = pd.read_csv('{}/data/ieee/train_identity.csv'.format(cwd), header=None, engine='python')
        test_transaction = pd.read_csv('{}/data/ieee/test_transaction.csv'.format(cwd), header=None, engine='python')
        test_identity = pd.read_csv('{}/data/ieee/test_identity.csv'.format(cwd), header=None, engine='python')

        # TODO X_train =
        # TODO y_train =
        # TODO X_test =
        # TODO y_test =
    else:
        raise ("Dataset not found")

    # Set methods
    methods = (
        # ('ABOD', ABOD()),
        # ('AutoEncoder', AutoEncoder(), SUPERVISED),
        # ('CBLOF', CBLOF()),
        # ('HBOS', HBOS()),
        # ('IForest', IForest()),
        ('KNN', KNN(), UNSUPERVISED),
        ('LOF', LOF(), UNSUPERVISED),
        ('OCSVM', OCSVM(), UNSUPERVISED),
        ('GaussianMixture',
         GaussianMixture(n_components=2, covariance_type='full', random_state=int(args.seed)),
         SEMI_SUPERVISED),
    )

    # Set k-folds
    skf = StratifiedKFold(n_splits=2, shuffle=False, random_state=None)

    # Plotting
    plt.style.use('dark_background')
    roc_auc_fig = plt.figure()
    roc_auc_ax = roc_auc_fig.subplots()
    roc_auc_ax.set_title('ROC-AUC Curve')

    cv_fig = plt.figure()
    cv_fig_ax = cv_fig.subplots()

    labels = ['Not Fraud', 'Fraud']

    # Running
    for name, method, level in methods:
        log(logfile, dataset_name + ", " + name)

        # Initialize metrics
        precision_cv_scores = []
        recall_cv_scores = []
        f2_cv_scores = []
        roc_cv_scores = []
        best_f2_threshold = []

        # Train
        if level == UNSUPERVISED:
            # Undersample majority 'Not Fraud' class only on training
            X_train_undersampled, _ = RandomUnderSampler(random_state=int(args.seed)).fit_resample(X_train_F_NF,
                                                                                                   y_train_F_NF)

            method.fit(X_train_undersampled)

        elif level == SEMI_SUPERVISED:
            # Only normal labels are known

            # Undersample majority 'Not Fraud' class only on training
            if name != 'GaussianMixture':
                X_train_undersampled, y_train_undersampled = RandomUnderSampler(random_state=int(args.seed)) \
                    .fit_resample(X_train_F_NF, y_train_F_NF)
                labels = np.copy(y_train_undersampled)
                labels[y_train_undersampled == FRAUD] = -1
                method.fit(X_train_undersampled, labels)
            else:
                method.fit(X_train_NF, y_train_NF)

        elif level == SUPERVISED:
            # Undersample majority 'Not Fraud' class only on training
            X_train_undersampled, y_train_undersampled = RandomUnderSampler(random_state=int(args.seed)) \
                .fit_resample(X_train_F_NF, y_train_F_NF)

            method.fit(X_train_undersampled, y_train_undersampled)
        else:
            raise ("Supervision level not found.")

        if name == 'GaussianMixture':
            for split_idx, (train_index, valid_index) in enumerate(skf.split(X_cv, y_cv)):
                X_cv_train, X_cv_valid = X_cv[train_index], X_cv[valid_index]
                y_cv_train, y_cv_valid = y_cv[train_index], y_cv[valid_index]

                # Estimate mu, sigma, generate multivariate normal variable
                mu = np.mean(X_cv_valid, axis=0)
                sigma = np.cov(X_cv_valid.T)
                mnv = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)

                # Understand possible threshold values
                if split_idx == 0:
                    not_fraud_logpdf = np.median(mnv.logpdf(X_cv[valid_index][y_cv[valid_index] == NOT_FRAUD]))
                    fraud_logpdf = np.median(mnv.logpdf(X_cv[valid_index][y_cv[valid_index] == FRAUD]))
                    log(logfile, "Not Fraud logpdf median: {}".format(not_fraud_logpdf))
                    log(logfile, "Fraud logpdf median: {}".format(fraud_logpdf))

                # Compute the weighted log probabilities for each sample
                y_valid_score_samples = method.score_samples(X_cv_valid)

                # Search best threshold
                thresholds = -np.arange(0, 1000, 2)

                scores = []
                for threshold in thresholds:
                    y_hat = (y_valid_score_samples < threshold).astype(int)
                    scores.append([recall_score(y_true=y_cv_valid, y_pred=y_hat),
                                   precision_score(y_true=y_cv_valid, y_pred=y_hat),
                                   fbeta_score(y_true=y_cv_valid, y_pred=y_hat, beta=2),
                                   roc_auc_score(y_true=y_cv_valid, y_score=y_hat)])

                scores = np.array(scores)
                best_threshold_index = scores[:, 2].argmax()
                best_threshold = thresholds[best_threshold_index]
                best_f2_threshold.append(best_threshold)

                # Plot gmm threshold
                cv_fig_ax.set(xticks=[best_threshold])
                cv_fig_ax.plot(thresholds, scores[:, 0], label='Recall')
                cv_fig_ax.plot(thresholds, scores[:, 1], label='Precision')
                cv_fig_ax.plot(thresholds, scores[:, 2], label='F2')
                cv_fig_ax.set_ylabel('Score')
                cv_fig_ax.set_xlabel('Threshold')
                cv_fig_ax.legend(loc='best')
                cv_fig_ax.figure.savefig(outdir + 'gmm_threshold_cv_sample_{}.png'.format(split_idx),
                                         bbox_inches='tight')
                cv_fig = plt.figure()
                cv_fig_ax = cv_fig.subplots()

                # Save Test Valid results
                recall_cv_scores.append(scores[best_threshold_index, 0])
                precision_cv_scores.append(scores[best_threshold_index, 1])
                f2_cv_scores.append(scores[best_threshold_index, 2])
                roc_cv_scores.append(scores[best_threshold_index, 3])

            # Average CV results
            log(logfile, '')
            log(logfile, "Avg CV Precision: {}"
                .format(np.round(np.mean(precision_cv_scores), 3)) + " +/- " + str(
                np.round(np.std(precision_cv_scores), 3)))
            log(logfile, "Avg CV  Recall: {}"
                .format(np.round(np.mean(recall_cv_scores), 3)) + " +/- " + str(np.round(np.std(recall_cv_scores), 3)))
            log(logfile, "Avg CV  F2: {}"
                .format(np.round(np.mean(f2_cv_scores), 3)) + " +/- " + str(np.round(np.std(f2_cv_scores), 3)))

            # Best threshold according to F2
            best_threshold = best_f2_threshold[np.array(f2_cv_scores).argmax()]

            # Test

            # Predict probabilities
            y_test_probs = (method.score_samples(X_test) < best_threshold).astype(int)
            log(logfile, classification_report(y_test, y_test_probs, target_names=labels))
            log(logfile, 'Best Threshold: %d' % best_threshold)

            log(logfile, 'Test Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_test_probs))
            log(logfile, 'Test Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_test_probs))
            log(logfile, 'Test F2: %.3f' % fbeta_score(y_true=y_test, y_pred=y_test_probs, beta=2))
            log(logfile, 'Test ROC AUC: %.3f' % roc_auc_score(y_test, y_test_probs))
            log(logfile, '')

            # Plot heatmap
            fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
            cm = pd.crosstab(y_test, y_test_probs, rownames=['Actual'],
                             colnames=['Predicted'])
            sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, ax=ax1, linewidths=.2, fmt='g')
            plt.title('Confusion Matrix for {}'.format(name), fontsize=14)
            plt.savefig(outdir + '{}_confusion_matrix.png'.format(name))
            plt.clf()
            plt.close(fig)

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_test_probs)
            roc_auc_ax.plot(fpr, tpr, marker='.', label=name)
        else:
            # Predict probabilities
            y_test_pred = method.predict(X_test)
            y_test_scores = method.decision_function(X_test)

            # Save results
            log(logfile, 'Test Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_test_pred))
            log(logfile, 'Test Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_test_pred))
            log(logfile, 'Test F2: %.3f' % fbeta_score(y_true=y_test, y_pred=y_test_pred, beta=2))
            log(logfile, 'Test ROC AUC: %.3f' % roc_auc_score(y_test, y_test_scores))
            log(logfile, '')

            # Plot heatmap
            fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
            cm = pd.crosstab(y_test, y_test_pred, rownames=['Actual'],
                             colnames=['Predicted'])
            sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, ax=ax1, linewidths=.2, fmt='g')
            plt.title('Confusion Matrix for {}'.format(name), fontsize=14)
            plt.savefig(outdir + '{}_confusion_matrix.png'.format(name))
            plt.clf()
            plt.close(fig)

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_test_scores)
            roc_auc_ax.plot(fpr, tpr, marker='x', label=name)

        log(logfile, '---' * 45)

    # Generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    # Plot the roc curve for the model
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    roc_auc_ax.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    roc_auc_ax.set_xlabel('False Positive Rate')
    roc_auc_ax.set_ylabel('True Positive Rate')
    roc_auc_ax.legend()
    roc_auc_ax.figure.savefig(outdir + 'roc_curve.png')


if __name__ == "__main__":
    experiments(config_file=sys.argv[1])
