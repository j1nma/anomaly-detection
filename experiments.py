import warnings

from pyod.models.iforest import IForest

warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import datetime
import gc
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras
from pyod.models.auto_encoder import AutoEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from matplotlib.patches import Ellipse
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report, fbeta_score, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

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
    parser.add_argument(
        "-fd",
        "--fraud_ratio",
        default=0.1,
        help="Desired ratio of fraud datapoints."
    )
    parser.add_argument(
        "-p",
        "--sampling",
        default="under",
        help="Sampling: under, over, smote"
    )

    return parser


####
# Function found at https://www.kaggle.com/jesucristo/fraud-complete-eda
####
def reduce_memory_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

                # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    # if mx < 255:
                    #     props[col] = props[col].astype(np.uint8)
                    # elif mx < 65535:
                    #     props[col] = props[col].astype(np.uint16)
                    if mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    # if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    #     props[col] = props[col].astype(np.int8)
                    # elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    #     props[col] = props[col].astype(np.int16)
                    if mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

                        # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")
        else:
            props[col] = props[col].fillna('NaN')

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")

    return props, NAlist


####
# Function found at https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
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

    w_factor = 0.75 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos[np.ix_([17, 14])], covar[np.ix_([17, 14], [17, 14])], alpha=w * w_factor)

    plt.savefig(outdir + 'gmm_v17_v14.png', bbox_inches='tight')
    plt.clf()

    return True


def group_emails(df_train, df_test):
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum',
              'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
              'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft',
              'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
              'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
              'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft',
              'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
              'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
              'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
              'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
              'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other',
              'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
              'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other',
              'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other',
              'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other',
              'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
              'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

    us_emails = ['gmail', 'net', 'edu']

    # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
    for c in ['P_emaildomain', 'R_emaildomain']:
        df_train[c + '_bin'] = df_train[c].map(emails)
        df_test[c + '_bin'] = df_test[c].map(emails)

        df_train[c + '_suffix'] = df_train[c].map(lambda x: str(x).split('.')[-1])
        df_test[c + '_suffix'] = df_test[c].map(lambda x: str(x).split('.')[-1])

        df_train[c + '_suffix'] = df_train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        df_test[c + '_suffix'] = df_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


###
# Feature Engineering : https://www.kaggle.com/artgor/eda-and-models#Feature-engineering
###
def feature_engineering(df_train, df_test, logfile):
    df_train['TransactionAmt_to_mean_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])[
        'TransactionAmt'].transform('mean')
    df_train['TransactionAmt_to_mean_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])[
        'TransactionAmt'].transform('mean')
    df_train['TransactionAmt_to_std_card1'] = df_train['TransactionAmt'] / df_train.groupby(['card1'])[
        'TransactionAmt'].transform('std')
    df_train['TransactionAmt_to_std_card4'] = df_train['TransactionAmt'] / df_train.groupby(['card4'])[
        'TransactionAmt'].transform('std')

    df_test['TransactionAmt_to_mean_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])[
        'TransactionAmt'].transform(
        'mean')
    df_test['TransactionAmt_to_mean_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])[
        'TransactionAmt'].transform(
        'mean')
    df_test['TransactionAmt_to_std_card1'] = df_test['TransactionAmt'] / df_test.groupby(['card1'])[
        'TransactionAmt'].transform(
        'std')
    df_test['TransactionAmt_to_std_card4'] = df_test['TransactionAmt'] / df_test.groupby(['card4'])[
        'TransactionAmt'].transform(
        'std')

    df_train['id_02_to_mean_card1'] = df_train['id_02'] / df_train.groupby(['card1'])['id_02'].transform('mean')
    df_train['id_02_to_mean_card4'] = df_train['id_02'] / df_train.groupby(['card4'])['id_02'].transform('mean')
    df_train['id_02_to_std_card1'] = df_train['id_02'] / df_train.groupby(['card1'])['id_02'].transform('std')
    df_train['id_02_to_std_card4'] = df_train['id_02'] / df_train.groupby(['card4'])['id_02'].transform('std')

    df_test['id_02_to_mean_card1'] = df_test['id_02'] / df_test.groupby(['card1'])['id_02'].transform('mean')
    df_test['id_02_to_mean_card4'] = df_test['id_02'] / df_test.groupby(['card4'])['id_02'].transform('mean')
    df_test['id_02_to_std_card1'] = df_test['id_02'] / df_test.groupby(['card1'])['id_02'].transform('std')
    df_test['id_02_to_std_card4'] = df_test['id_02'] / df_test.groupby(['card4'])['id_02'].transform('std')

    df_train['D15_to_mean_card1'] = df_train['D15'] / df_train.groupby(['card1'])['D15'].transform('mean')
    df_train['D15_to_mean_card4'] = df_train['D15'] / df_train.groupby(['card4'])['D15'].transform('mean')
    df_train['D15_to_std_card1'] = df_train['D15'] / df_train.groupby(['card1'])['D15'].transform('std')
    df_train['D15_to_std_card4'] = df_train['D15'] / df_train.groupby(['card4'])['D15'].transform('std')

    df_test['D15_to_mean_card1'] = df_test['D15'] / df_test.groupby(['card1'])['D15'].transform('mean')
    df_test['D15_to_mean_card4'] = df_test['D15'] / df_test.groupby(['card4'])['D15'].transform('mean')
    df_test['D15_to_std_card1'] = df_test['D15'] / df_test.groupby(['card1'])['D15'].transform('std')
    df_test['D15_to_std_card4'] = df_test['D15'] / df_test.groupby(['card4'])['D15'].transform('std')

    df_train['D15_to_mean_addr1'] = df_train['D15'] / df_train.groupby(['addr1'])['D15'].transform('mean')
    df_train['D15_to_mean_addr2'] = df_train['D15'] / df_train.groupby(['addr2'])['D15'].transform('mean')
    df_train['D15_to_std_addr1'] = df_train['D15'] / df_train.groupby(['addr1'])['D15'].transform('std')
    df_train['D15_to_std_addr2'] = df_train['D15'] / df_train.groupby(['addr2'])['D15'].transform('std')

    df_test['D15_to_mean_addr1'] = df_test['D15'] / df_test.groupby(['addr1'])['D15'].transform('mean')
    df_test['D15_to_mean_addr2'] = df_test['D15'] / df_test.groupby(['addr2'])['D15'].transform('mean')
    df_test['D15_to_std_addr1'] = df_test['D15'] / df_test.groupby(['addr1'])['D15'].transform('std')
    df_test['D15_to_std_addr2'] = df_test['D15'] / df_test.groupby(['addr2'])['D15'].transform('std')

    df_train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = df_train['P_emaildomain'].str.split('.',
                                                                                                              expand=True)
    df_train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = df_train['R_emaildomain'].str.split('.',
                                                                                                              expand=True)
    df_test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = df_test['P_emaildomain'].str.split('.',
                                                                                                            expand=True)
    df_test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = df_test['R_emaildomain'].str.split('.',
                                                                                                            expand=True)

    one_value_cols_train = [col for col in df_train.columns if df_train[col].nunique() <= 1]
    many_null_cols_train = [col for col in df_train.columns if
                            df_train[col].isnull().sum() / df_train.shape[0] > 0.9]
    big_top_value_cols_train = [col for col in df_train.columns if
                                df_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    one_value_cols_test = [col for col in df_test.columns if df_test[col].nunique() <= 1]
    many_null_cols_test = [col for col in df_test.columns if df_test[col].isnull().sum() / df_test.shape[0] > 0.9]
    big_top_value_cols_test = [col for col in df_test.columns if
                               df_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    cols_to_drop = list(set(
        many_null_cols_train + many_null_cols_test + big_top_value_cols_train + big_top_value_cols_test + one_value_cols_train + one_value_cols_test))
    cols_to_drop.remove('isFraud')
    log(logfile, "Number of columns to drop: " + str(len(cols_to_drop)))
    log(logfile, "Columns to drop: " + str(cols_to_drop))

    df_train = df_train.sort_values('TransactionDT').drop(cols_to_drop, axis=1, errors='ignore')
    df_test = df_test.sort_values('TransactionDT').drop(cols_to_drop, axis=1, errors='ignore')
    return df_train, df_test


def experiments(config_file):
    warnings.filterwarnings('ignore')

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

        # Read data
        df = pd.read_csv('{}/data/creditcard/creditcard.csv'.format(cwd))

        # Normalize 'Amount' column
        df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

        # Normalize 'Time' column
        df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

        # Set features and labels
        X = df.drop(['Class'], axis=1)
        y = df['Class']

        # Split dataframe by class
        df_NF = df[df.Class == NOT_FRAUD]
        df_F = df[df.Class == FRAUD]

        # Split by data and labels
        X_NF, X_F = df_NF.drop(['Class'], axis=1), df_F.drop(['Class'], axis=1)
        y_NF, y_F = df_NF['Class'], df_F['Class']
        contamination = len(y_F) / (len(y_F) + len(y_NF))

        ### SUPERVISED, UNSUPERVISED ###
        # Split 80-20
        X_train_F_NF, X_test_F_NF, y_train_F_NF, y_test_F_NF = train_test_split(X, y,
                                                                                test_size=0.5,  # 0.5 for OCSVM
                                                                                random_state=int(args.seed))

        ### SEMI-SUPERVISED ###
        # Split 80-20 NF
        X_train_NF, X_cv_test_NF, y_train_NF, y_cv_test_NF = train_test_split(X_NF, y_NF,
                                                                              test_size=0.2,
                                                                              random_state=int(args.seed))

        # Split extra NF for cross validation and testing
        X_cv_NF, X_test_NF, y_cv_NF, y_test_NF = train_test_split(X_cv_test_NF, y_cv_test_NF,
                                                                  test_size=0.5,
                                                                  random_state=int(args.seed))

        # Split 50-50 F data
        X_cv_F, X_test_F, y_cv_F, y_test_F = train_test_split(X_F, y_F, test_size=0.5, random_state=int(args.seed))

        # Build threshold cross validation and testing sets
        X_cv = np.vstack([X_cv_NF, X_cv_F])
        y_cv = np.hstack([y_cv_NF, y_cv_F])
        X_test = np.vstack([X_test_NF, X_test_F])
        y_test = np.hstack([y_test_NF, y_test_F])
        X_test_df = pd.concat([X_test_NF, X_test_F])

    elif str(args.dataset) == 'ieee':
        # Read data
        dataset_name = 'ieee'

        df_train_transaction = pd.read_csv('{}/data/ieee/train_transaction.csv'.format(cwd), index_col='TransactionID')
        # nrows=n_rows)
        df_train_identity = pd.read_csv('{}/data/ieee/train_identity.csv'.format(cwd), index_col='TransactionID')
        # nrows=n_rows)
        df_test_transaction = pd.read_csv('{}/data/ieee/test_transaction.csv'.format(cwd), index_col='TransactionID')
        # nrows=n_rows)
        df_test_identity = pd.read_csv('{}/data/ieee/test_identity.csv'.format(cwd), index_col='TransactionID')
        # nrows=n_rows)
        df_submission = pd.read_csv('{}/data/ieee/sample_submission.csv'.format(cwd), index_col='TransactionID')
        # nrows=n_rows)

        # Match both columns
        fix = {o: n for o, n in zip(df_test_identity.columns, df_train_identity.columns)}
        df_test_identity.rename(columns=fix, inplace=True)

        # Merge and set train, test
        df_train = df_train_transaction.merge(df_train_identity, how='left', left_index=True, right_index=True,
                                              on='TransactionID')
        df_test = df_test_transaction.merge(df_test_identity, how='left', left_index=True, right_index=True,
                                            on='TransactionID')

        log(logfile, f'There are {df_train.isnull().any().sum()} columns in train dataset with missing values.')

        log(logfile, 'Train shape: ' + str(df_train.shape) + ', Test shape: ' + str(df_test.shape))

        # Group emails: https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
        group_emails(df_train, df_test)

        # Normalize D columns: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
        for i in range(1, 16):
            if i in [1, 2, 3, 5, 9]: continue
            df_train['D' + str(i)] = df_train['D' + str(i)] - df_train.TransactionDT / np.float32(24 * 60 * 60)
            df_test['D' + str(i)] = df_test['D' + str(i)] - df_test.TransactionDT / np.float32(24 * 60 * 60)

        # Feature engineering
        df_train, df_test = feature_engineering(df_train, df_test, logfile)

        # Cleaning infinite values to NaN by https://www.kaggle.com/dimartinot
        df_train = df_train.replace([np.inf, -np.inf], np.nan)
        df_test = df_test.replace([np.inf, -np.inf], np.nan)

        # Reduce memory usage
        df_train, _ = reduce_memory_usage(df_train)
        df_test, _ = reduce_memory_usage(df_test)

        # Encoding categorical features
        for f in df_train.drop('isFraud', axis=1).columns:
            if str(df_train[f].dtype) == 'object' or str(df_test[f].dtype) == 'object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(df_train[f].values) + list(df_test[f].values))
                df_train[f] = lbl.transform(list(df_train[f].values))
                df_test[f] = lbl.transform(list(df_test[f].values))

        # Free objects
        del df_train_transaction, df_train_identity, df_test_transaction, df_test_identity

        print('Train shape', df_train.shape, 'test shape', df_test.shape)

        ### SUPERVISED ###
        # Set features and labels
        X_test = df_test.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1, errors='ignore')
        X_test_df = X_test  # For compatibility with other if clauses at Testing stage

        # Split dataframe by class
        df_train_NF = df_train[df_train.isFraud == NOT_FRAUD]
        df_train_F = df_train[df_train.isFraud == FRAUD]

        # Free objects
        del df_train, df_test

        # Collect garbage
        gc.collect()

        # Split by data and labels
        X_train_NF = df_train_NF.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1, errors='ignore')
        X_train_F = df_train_F.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1, errors='ignore')

        y_train_NF, y_train_F = df_train_NF['isFraud'], df_train_F['isFraud']
        contamination = len(y_train_F) / (len(y_train_F) + len(y_train_NF))

        ### SEMI-SUPERVISED ###
        # Split 80-20 NF for cross validation and testing
        X_train_NF, X_cv_NF, y_train_NF, y_cv_NF = train_test_split(X_train_NF, y_train_NF,
                                                                    test_size=0.5,
                                                                    random_state=int(args.seed))

        # Split 50-50 F data
        X_train_F, X_cv_F, y_train_F, y_cv_F = train_test_split(X_train_F, y_train_F, test_size=0.5,
                                                                random_state=int(args.seed))

        # Build cross validation and testing sets
        # X_train_F_NF = np.vstack([X_train_NF, X_train_F])
        X_train_F_NF = pd.concat([X_train_NF, X_train_F])  # for XGBoost works with df
        # y_train_F_NF = np.hstack([y_train_NF, y_train_F])
        y_train_F_NF = pd.concat([y_train_NF, y_train_F])
        X_cv = np.vstack([X_cv_NF, X_cv_F])
        y_cv = np.hstack([y_cv_NF, y_cv_F])
    else:
        raise ("Dataset not found")

    # Set methods
    methods = (
        ('IForest', IForest(contamination=contamination, n_jobs=-1, random_state=int(args.seed), verbose=1),
         UNSUPERVISED),
        ('LOF', LOF(n_neighbors=3, contamination=contamination, n_jobs=-1), UNSUPERVISED),
        ('OCSVM', OCSVM(kernel='linear', contamination=contamination, tol=.1, verbose=True, cache_size=2000),
         UNSUPERVISED),
        ('GaussianMixture',
         GaussianMixture(n_components=2, covariance_type='full', random_state=int(args.seed)),
         SEMI_SUPERVISED),
        ('XGBClassifier', XGBClassifier(max_depth=9, n_jobs=-1, verbosity=3, random_state=int(args.seed)), SUPERVISED),
    )

    methods = list(methods)
    if str(args.dataset) == 'creditcard':
        methods.insert(0, ('AutoEncoder',
                           AutoEncoder(hidden_neurons=[64, 30, 30, 64], verbose=2, epochs=70, batch_size=320,
                                       random_state=int(args.seed),
                                       contamination=contamination), UNSUPERVISED), )
    else:
        methods.insert(0, ('AutoEncoder',
                           AutoEncoder(hidden_neurons=[64, 30, 30, 64], verbose=2, epochs=35, batch_size=320,
                                       random_state=int(args.seed),
                                       contamination=contamination), UNSUPERVISED), )
    methods = tuple(methods)

    # Get fraud-ratio
    fraud_ratio = float(args.fraud_ratio)

    # Set k-folds
    skf = StratifiedKFold(n_splits=3, shuffle=False, random_state=None)

    # Plotting
    plt.style.use('dark_background')
    roc_auc_fig = plt.figure()
    roc_auc_ax = roc_auc_fig.subplots()
    roc_auc_ax.set_title('ROC-AUC Curve')

    cv_fig = plt.figure()
    cv_fig_ax = cv_fig.subplots()

    labels = ['Not Fraud', 'Fraud']

    # Set sampler
    if str(args.sampling) == 'under':
        sampler = RandomUnderSampler(sampling_strategy=fraud_ratio, random_state=int(args.seed))
    elif str(args.sampling) == 'smote':
        sampler = SMOTE(sampling_strategy=fraud_ratio, n_jobs=-1, random_state=int(args.seed))
    elif str(args.sampling) == 'over':
        sampler = RandomOverSampler(random_state=int(args.seed))
    else:
        raise ("Sampling method not found.")

    # Running
    for method_name, method, level in methods:
        log(logfile, dataset_name + ", " + method_name)

        # Train
        if level == SEMI_SUPERVISED:
            # Only normal labels are known
            # No 'y' needed, because it is known that X is NF
            method.fit(X_train_NF)

        elif level == UNSUPERVISED or level == SUPERVISED:

            # Initialize CV metrics
            precision_cv_scores = []
            recall_cv_scores = []
            f2_cv_scores = []
            roc_cv_scores = []

            # Cross validate while sampling
            for split_idx, (train_index, valid_index) in enumerate(skf.split(X_train_F_NF, y_train_F_NF)):
                X_fold_train, X_fold_valid = X_train_F_NF.iloc[train_index], X_train_F_NF.iloc[valid_index]
                y_fold_train, y_fold_valid = y_train_F_NF.iloc[train_index], y_train_F_NF.iloc[valid_index]

                # Sample
                X_fold_train_resampled, y_fold_train_resampled = sampler.fit_resample(X_fold_train, y_fold_train)

                # plot_gmm(GaussianMixture(n_components=2, covariance_type='full', random_state=int(args.seed)),
                #          X_fold_train_resampled.sample(frac=0.5),
                #          outdir=outdir)

                if method_name == 'LOF' or method_name == 'OCSVM':
                    X_fold_train_resampled, y_fold_train_resampled = X_fold_train_resampled.sample(
                        frac=0.1), y_fold_train_resampled.sample(frac=0.1)  # for few CPUs

                # Fit
                if level == UNSUPERVISED:
                    method.fit(X_fold_train_resampled)
                elif level == SUPERVISED:
                    method.fit(X_fold_train_resampled, y_fold_train_resampled)

                # Validate
                y_fold_pred = method.predict(X_fold_valid)
                if method_name == 'XGBClassifier':
                    y_fold_scores = method.predict_proba(X_fold_valid)[:, 1]
                else:
                    y_fold_scores = method.decision_function(np.array(X_fold_valid))

                # Save fold results
                precision_cv_scores.append(precision_score(y_true=y_fold_valid, y_pred=y_fold_pred))
                recall_cv_scores.append(recall_score(y_true=y_fold_valid, y_pred=y_fold_pred))
                f2_cv_scores.append(fbeta_score(y_true=y_fold_valid, y_pred=y_fold_pred, beta=2))
                roc_cv_scores.append(roc_auc_score(y_true=y_fold_valid, y_score=y_fold_scores))
                log(logfile, "Fold Precision: {}".format(str(np.round(precision_cv_scores[-1], 3))))
                log(logfile, "Fold Recall: {}".format(str(np.round(recall_cv_scores[-1], 3))))
                log(logfile, "Fold F2: {}".format(str(np.round(f2_cv_scores[-1], 3))))
                log(logfile, "Fold ROC-AUC: {}".format(str(np.round(roc_cv_scores[-1], 3))))

            # Average Training CV results
            log(logfile, '')
            log(logfile, "Avg Training CV Precision: {}"
                .format(np.round(np.mean(precision_cv_scores), 3)) + " +/- " + str(
                np.round(np.std(precision_cv_scores), 3)))
            log(logfile, "Avg Training CV Recall: {}"
                .format(np.round(np.mean(recall_cv_scores), 3)) + " +/- " + str(
                np.round(np.std(recall_cv_scores), 3)))
            log(logfile, "Avg Training CV F2: {}"
                .format(np.round(np.mean(f2_cv_scores), 3)) + " +/- " + str(np.round(np.std(f2_cv_scores), 3)))
            log(logfile, "Avg Training CV ROC-AUC: {}"
                .format(np.round(np.mean(roc_cv_scores), 3)) + " +/- " + str(
                np.round(np.std(roc_cv_scores), 3)))

        else:
            raise ("Supervision level not found.")

        # TEST

        # Re-initialize metrics
        precision_cv_scores = []
        recall_cv_scores = []
        f2_cv_scores = []
        roc_cv_scores = []
        best_f2_threshold = []

        if method_name == 'GaussianMixture':

            # Threshold search: Cross validate
            for split_idx, (_, valid_index) in enumerate(skf.split(X_cv, y_cv)):
                X_cv_valid = X_cv[valid_index]
                y_cv_valid = y_cv[valid_index]

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

            # Best threshold according to F2
            best_threshold = best_f2_threshold[np.array(f2_cv_scores).argmax()]
            log(logfile, 'Best Threshold: %d' % best_threshold)

            # Average CV results
            log(logfile, '')
            log(logfile, "Avg Threshold CV Precision: {}"
                .format(np.round(np.mean(precision_cv_scores), 3)) + " +/- " + str(
                np.round(np.std(precision_cv_scores), 3)))
            log(logfile, "Avg Threshold CV Recall: {}"
                .format(np.round(np.mean(recall_cv_scores), 3)) + " +/- " + str(np.round(np.std(recall_cv_scores), 3)))
            log(logfile, "Avg Threshold CV F2: {}"
                .format(np.round(np.mean(f2_cv_scores), 3)) + " +/- " + str(np.round(np.std(f2_cv_scores), 3)))
            log(logfile, "Avg Threshold CV ROC-AUC: {}"
                .format(np.round(np.mean(roc_cv_scores), 3)) + " +/- " + str(np.round(np.std(roc_cv_scores), 3)))

            # Test
            # Predict probabilities
            y_test_pred = (method.score_samples(X_test) < best_threshold).astype(int)
            y_test_scores = y_test_pred  # The same for Gaussian ROC-AUC
            y_test_fraud_probabilities = method.predict_proba(X_test)[:, 1]

        else:
            # Test
            # Predict probabilities
            if method_name == 'XGBClassifier':
                y_test_pred = method.predict(X_test_df)
                # y_test_scores = y_test_pred
                y_test_fraud_probabilities = method.predict_proba(X_test_df)[:, 1]
                y_test_scores = y_test_fraud_probabilities
            else:
                # y_test_pred = method.predict(X_test_df)
                # y_test_fraud_probabilities = method.predict_proba(X_test_df)[:, 1]
                # y_test_scores = method.decision_function(X_test_df)

                batch_size = 20000  # chunk row size
                y_test_pred_list_df = []
                y_test_fraud_probabilities_list_df = []
                y_test_scores_list_df = []
                for batch_number, X_test_batch in X_test_df.groupby(np.arange(len(X_test_df)) // batch_size):
                    y_test_pred_list_df.append(method.predict(X_test_batch))
                    y_test_fraud_probabilities_list_df.append(method.predict_proba(X_test_batch)[:, 1])
                    y_test_scores_list_df.append(method.decision_function(X_test_batch))
                y_test_fraud_probabilities = np.hstack(y_test_fraud_probabilities_list_df)

        if dataset_name != 'ieee':
            # Plot heatmap
            fig, (ax1) = plt.subplots(ncols=1, figsize=(5, 5))
            cm = pd.crosstab(y_test, y_test_pred, rownames=['Actual'],
                             colnames=['Predicted'])
            sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, ax=ax1, linewidths=.2, fmt='g')
            plt.title('Confusion Matrix for {}'.format(method_name), fontsize=14)
            plt.savefig(outdir + '{}_confusion_matrix.png'.format(method_name))
            plt.clf()
            plt.close(fig)

        # Save results
        if dataset_name == 'ieee':
            df_submission['isFraud'] = y_test_fraud_probabilities
            df_submission.to_csv(outdir + '{}_{}_ieee_submission.csv'.format(method_name, str(args.sampling)))
        else:
            log(logfile, classification_report(y_test, y_test_pred, target_names=labels))
            log(logfile, 'Test Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_test_pred))
            log(logfile, 'Test Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_test_pred))
            log(logfile, 'Test F2: %.3f' % fbeta_score(y_true=y_test, y_pred=y_test_pred, beta=2))
            log(logfile, 'Test ROC AUC: %.3f' % roc_auc_score(y_test, y_test_scores))
            log(logfile, '')

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_test_scores)
            roc_auc_ax.plot(fpr, tpr, marker='x', label=method_name)

        log(logfile, '---' * 45)

    if dataset_name != 'ieee':
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
