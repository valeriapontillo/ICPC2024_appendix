import numpy as np
import pandas as pd
import json

from scipy.stats import uniform
from sklearn.model_selection import LeaveOneGroupOut, TimeSeriesSplit, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, accuracy_score, \
    f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

################################################################
#
#        Hyperparameters Optimization
#
################################################################

optimization_available = ["none", "randomsearch"]


def parameters_space(model):
    if model == 'randomforest':
        n_estimators = [10, 50, 100, 150, 200]
        max_features = ["sqrt", "log2"]
        # max_depth = list(np.arange(3, 6, 12, 24)) + [None]
        min_samples_split = [2, 3, 4, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        criterion = ["gini", "entropy"]

        param_space = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            # "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap,
            "criterion": criterion
        }

    if model == 'adaboost':
        n_estimators = np.arange(100, 1000, step=100)
        learning_rate = [0.001, 0.01, 0.1]

        param_space = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate
        }

    if model == 'multilayerperceptron':
        hidden_layers_size = [(10, 30, 10), (20,)]
        activation = ['tanh', 'relu']
        solver = ['sgd', 'adam']
        alpha = [0.0001, 0.05]
        learning_rate = ['constant', 'adaptive']

        param_space = {
            "hidden_layer_sizes": hidden_layers_size,
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "learning_rate": learning_rate
        }

    if model == 'decisiontree':
        criterion = ['gini']
        max_features = ["auto"]
        max_depth = list(np.arange(10, 100, step=10)) + [None]
        min_samples_split = np.arange(2, 10, step=2)
        min_samples_leaf = [1, 2]

        param_space = {
            "criterion": criterion,
            "max_features": max_features,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }

    if model == 'svm':
        dual = [True, False]
        C = uniform(1, 10)

        param_space = {
            "dual": dual,
            "C": C
        }

    if model == 'knn':
        leaf_size = list(range(1, 50))
        n_neighbors = list(range(1, 30))
        p = [1, 2]

        param_space = {
            "leaf_size": leaf_size,
            "n_neighbors": n_neighbors,
            "p": p
        }

    if model == "naivebayes":
        smoothing = np.logspace(1, 5, num=50)
        param_space = {
            'var_smoothing': smoothing
        }

    if model == "logisticregression":
        C = np.logspace(-4, 4)
        penalty = ['none', 'l2']
        param_space = {
            "C": C,
            "penalty": penalty
        }

    return param_space


def hyperparam_opt(clf, clf_name, opt_method, X, y):
    if opt_method == "randomsearch":
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        splits_indices = cv.split(X, y)

        random_search = RandomizedSearchCV(clf, parameters_space(clf_name), n_iter=10, cv=splits_indices, scoring="f1",
                                           n_jobs=-1, random_state=0, verbose=0)

        search = random_search.fit(X, y)

        best_param = search.best_params_
        best_score = search.best_score_
        return best_param


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# with open(filepath + ".pkl", 'wb') as f:
#   pickle.dump(best_params, f)

def save_best_params(best_params, filepath):
    with open(filepath + ".json", 'w') as f:
        to_dump = {}
        for k in best_params.keys():
            to_dump[k] = best_params[k]
        f.write(json.dumps(to_dump, cls=NpEncoder))


################################################################
#
#        Validation
#
################################################################

def walk_forward_release(X, y, releases):
    """
    Generate train and test splits fro TimeSeriesSplit on releases.
    Train consists of a release or a list of successive releases, and
    the test set consist of the next release in time
    :param X: array-like of shape (n_samples, m_features)
    :param y: array-like of shape (n_samples,)
    :param releases : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.
        Must be a list of integer, i.e., [1, 1, 1, 2, 2, 3, 4, 4, etc.].
        Each integer denotes a release. Files within the same release have the same group id.
    """
    X, _, releases = indexable(X, y, releases)
    n_samples = _num_samples(X)
    n_folds = len(set(releases))  # Number of distinct groups (releases)

    if n_folds > n_samples:
        raise ValueError(f"Cannot have number of folds ={n_folds} greater than the number of samples: {n_samples}.")

    indices = np.arange(n_samples)
    offset = 0

    for _ in range(0, n_folds - 1):
        try:
            train_indices = [i for i, x in enumerate(releases) if x == releases[offset]]
            offset += len(train_indices)

            test_indices = [j for j, y in enumerate(releases) if y == releases[offset]]

            yield indices[:offset], indices[offset: offset + len(test_indices)]
        except IndexError:
            print("train_indices: ")
            print(train_indices)
            print("offset: ")
            print(offset)
            print("test_indices: ")
            print(test_indices)


################################################################
#
#        Classifiers
#
################################################################

classifiers_available = ["naivebayes", "adaboost", "randomforest", "svm", "decisiontree", "multilayerperceptron",
                         "dummyrandom", "extratree", "logisticregression"]


def get_clf(param):
    if param == "svm":
        return LinearSVC(verbose=0, dual='auto')
    elif param == "decisiontree":
        return DecisionTreeClassifier(random_state=0)
    elif param == "adaboost":
        return AdaBoostClassifier(n_estimators=100, random_state=0)
    elif param == "multilayerperceptron":
        return MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                             random_state=1)
    elif param == "randomforest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, verbose=0)
    elif param == "knn":
        return KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    elif param == "naivebayes":
        return GaussianNB()
    elif param == "logisticregression":
        return LogisticRegression()
    elif param == "dummyopt":
        # always predict as non-exploitable
        return DummyClassifier(strategy="constant", constant=0)
    elif param == "dummypes":
        # always predict as exploitable
        return DummyClassifier(strategy="constant", constant=1)
    elif param == "dummyrandom":
        return DummyClassifier(strategy="uniform")
    elif param == "extratree":
        return ExtraTreesClassifier(random_state=42)


def scorer(clf, clf_name, X, y):
    """Evaluates the classifier and returns performance metrics."""

    # Ensure y is binary (just in case)
    y = np.array(y).astype(int)

    # Generate predictions
    y_pred = clf.predict(X)

    # Ensure y_pred is binary
    y_pred = np.round(y_pred).astype(int)

    cm = confusion_matrix(y, y_pred)

    try:
        tn, fp, fn, tp = cm.ravel()

        inspection_rate = (tp + fp) / (tp + tn + fp + fn)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
    except ValueError:
        tn = fp = fn = tp = np.nan
        inspection_rate = precision = recall = accuracy = f1 = mcc = np.nan

    # Compute ROC curve and AUC
    try:
        if clf_name == "svm":
            # Ensure SVM uses decision function properly
            probs = clf.decision_function(X)

            # Fix potential NaN issues by checking min and max values
            if np.isnan(probs).any():
                raise ValueError("NaN detected in decision_function output!")

            if np.max(probs) != np.min(probs):  # Avoid division by zero
                probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
            else:
                probs = np.zeros_like(probs)  # Assign zero probability if all values are identical

        else:
            probs = clf.predict_proba(X)[:, 1]

        if np.isnan(probs).any():
            raise ValueError("NaN detected in probability outputs!")

        fpr, tpr, thresholds = roc_curve(y, probs)
        auc_roc = roc_auc_score(y, probs)

    except (ValueError, IndexError):
        fpr, tpr, thresholds = [np.nan], [np.nan], [np.nan]
        auc_roc = 0

    # Store results in a DataFrame
    res = {
        "tp": [tp], "fp": [fp], "tn": [tn], "fn": [fn],
        "precision": [precision], "recall": [recall], "accuracy": [accuracy],
        "inspection_rate": [inspection_rate], "f1_score": [f1], "mcc": [mcc],
        "auc_roc": [auc_roc]
    }

    return fpr, tpr, pd.DataFrame(res), y_pred