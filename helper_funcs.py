import numpy as np
import pandas as pd


"""
A simple set of helper functions to make life a little easier :)

"""

def is_numpy(x):
    """
    As it states is the object x a numpy array
    """
    return isinstance(x, np.ndarray)


def is_pandas(x):
    """
        As it states is the object x a pandas array
    """
    return isinstance(x, pd.DataFrame)


def export_fold(train, folds, filename='./input/xfolds.csv'):
    """

    NEED TO DO MORE CHECKS TO MAKE SURE THE SPLITS ARE AS EXPECTED

    A simple helper function to export the folds created from sklearn.model_selection
    Its main purpose is to enable the export of the fold ids so stacking with different models, and languages
    and ensure that no leaking occurs.
    :param train: A pandas DF with an index ID
    :param folds: An sklearn.model_selection fold generator (KFold, StratifiedKFold, etc)
    :param filename: The path and file name to export the folds
    :return: Nothing, writes file to disk.
    """
    train['fold'] = 0
    # Capture the index
    i = 0
    for _, xtest in folds.split(train.index):
        print "Fold", i
        train['fold'].ix[xtest] = i
        i += 1
    print train.groupby('fold').count()
    # Export in case we want to use R
    train.to_csv(filename, header=True, index=None)

