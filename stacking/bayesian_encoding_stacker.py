import numpy as np
import pandas as pd


class StackingRegressor():
    """
    To facilitate stacking of regression models
    It only provides fit and predict functions, and works with a continuous target .
    :param base_regressors: A list of regression models with a fit and predict method similar to that of sklearn
    :param xfolds: A Kfold or KfoldStratified object to split the data for stacking.
    :param evaluation: optional evaluation metric (y_true, y_score) to check metric at each fold.
                    expected use case might be evaluation=from sklearn.metrics.mean_squared_error
    """

    def __init__(self, base_regressors, xfolds, evaluation=None):
        self.base_regressors = base_regressors
        self.xfolds = xfolds
        self.evaluation = evaluation

        # Build an empty pandas dataframe to store the meta results to.
        # As many rows as the folds data, as many cols as base regressors
        self.colnames = ["v" + str(n) for n in range(len(self.base_regressors))]
        self.stacking_train = None

    def fit(self, X, y, **kwargs):
        """ A generic fit method for meta stacking.
        :param X: Train dataset
        :param y: Train target
        :param kwargs: Any optional params to give the fit method, i.e in xgboost we may use eval_metirc='rmse'
        """
        self.stacking_train = pd.DataFrame(np.nan, index=X.index, columns=self.colnames)
        for model_no in range(len(self.base_regressors)):
            print "Running Model ", model_no + 1, "of", len(self.base_regressors)
            for traincv, testcv in self.xfolds:
                # Loop over the different folds.
                self.base_regressors[model_no].fit(X.ix[traincv], y.ix[traincv], **kwargs)
                predicted_y = self.base_regressors[model_no].predict(X.ix[testcv])
                if self.evaluation is not None:
                    print "Current Score = ", self.evaluation(y.ix[testcv], predicted_y)
                self.stacking_train.ix[testcv, model_no] = predicted_y
            # Finally fit against all the data
            self.base_regressors[model_no].fit(X, y, **kwargs)

    def predict(self, X):
        """
        :param X: The data to apply the fitted model from fit
        :return: The predicted value of the regression model
        """
        stacking_predict_data = pd.DataFrame(np.nan, index=X.index, columns=self.colnames)

        for model_no in range(len(self.base_regressors)):
            stacking_predict_data.ix[:, model_no] = self.base_regressors[model_no].predict(X)
        return stacking_predict_data

    @property
    def meta_train(self):
        """
        A return method for the underlying meta data prediction from the training data set as a pandas dataframe
        Use the predict method to score new data for each classifier.
        :return: A pandas dataframe object of stacked predictions of predict
        """
        return self.stacking_train.copy()