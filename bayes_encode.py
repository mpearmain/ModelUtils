from itertools import combinations
import pandas as pd


def bayesian_encode(train, test, encode_list, target_col, levels=1):
    """
    A generic function to encode the mean, median, and the stds of categorical variables against a target col.
    This can be used for both classification and regression problems

    :param train: The training data on which to select the vars to encode
    :param test: The training with the same cols as the training data
    :param encode_list: A list of colnames to encode
    :param levels: int, The depth of encoding i.e all singles, pairwise, triples
    :param target_col: The target col to encode against.
    :return: Train and test sets with encoded mean, median, and std's
    """

    # Build a list of encoders based on the combination levels.
    encode_combinations = list(combinations(encode_list, levels))

    # iterate through the combinations, first extract teh cols from the tuple.
    for encode_row in encode_combinations:
        encode_cols = []
        [encode_cols.append(j) for j in encode_row]

        aggr_funcs = ["mean", "median", "std"]
        global_mean = train[encode_cols].mean().mean()
        global_median = train[encode_cols].median().median()

        meanDF = pd.DataFrame(train.groupby(encode_cols)[target_col].aggregate(aggr_funcs))
        meanDF = meanDF.reset_index()
        # Take the mean of STD for missing
        global_std = meanDF['std'].mean()

        meanDF['mean'].fillna(global_mean, inplace=True)
        meanDF['median'].fillna(global_median, inplace=True)
        meanDF['std'].fillna(global_std, inplace=True)

        label = ""
        for i in encode_cols:
            label = label + i
        print "Getting", label, "wise demand.."

        dfcols = [[col for col in encode_cols], [i + label for i in aggr_funcs]]
        meanDF.columns = [item for sublist in dfcols for item in sublist]

        train = pd.merge(train, meanDF, on=encode_cols, how="left")
        test = pd.merge(test, meanDF, on=encode_cols, how="left")
        # fill any missing values (in test not in train)

        test['mean' + label].fillna(global_mean, inplace=True)
        test['median' + label].fillna(global_median, inplace=True)
        test['std' + label].fillna(global_std, inplace=True)
    return train, test
