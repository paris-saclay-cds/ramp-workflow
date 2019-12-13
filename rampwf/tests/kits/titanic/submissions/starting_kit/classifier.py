from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def get_classifier():
    clf = Pipeline([('classifier', LogisticRegression(C=1., solver='lbfgs'))])
    return clf
