
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.metrics import roc_auc_score, confusion_matrix
from python.utils import BinaryStackingClassifier


## settings
projPath = '/Users/michael.pearmain/tmp/emirates/'

## data
# read the training and test sets
X = pd.read_csv(projPath + 'input/xtrain.csv')
y = X.LEGTYPE
X.drop('LEGTYPE', axis=1, inplace=True)
X.drop('RowNumberID', axis=1, inplace=True)

xtest = pd.read_csv(projPath + 'input/xtest.csv')
ytest = xtest.LEGTYPE
xtest.drop('LEGTYPE', axis=1, inplace=True)
xtest.drop('RowNumberID', axis=1, inplace=True)

# folds
xfolds = pd.read_csv(projPath + 'input/xfolds.csv', )
xfolds.drop('valid', axis=1, inplace=True)


#use ONEGO technique to create stacking model
stacking_classifier = BinaryStackingClassifier(
    base_classifiers=[
        ExtraTreesClassifier(n_estimators=250, random_state=0, min_samples_leaf=5),
        RandomForestClassifier(n_estimators=250, random_state=0, min_samples_leaf=5),
        GradientBoostingClassifier(n_estimators=250, random_state=0,min_samples_leaf=5),
        NB()
    ],
    xfolds=xfolds,
    evaluation=roc_auc_score)

stacking_classifier.fit(X, y)
predicted_y_proba = stacking_classifier.predict_proba(xtest)

# Fit a linear regression to the meta level data

LR = LogisticRegressionCV(cv=5, random_state=0)
LR.fit(X=stacking_classifier.meta_train, y=y)
final_pred_train = LR.predict(stacking_classifier.meta_train)
final_pred_test = LR.predict(predicted_y_proba)

confusion_matrix(y, final_pred_train)
confusion_matrix(ytest, final_pred_test)


