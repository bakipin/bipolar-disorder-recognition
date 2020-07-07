from sklearn.svm import SVC
from sklearn.metrics import recall_score
from collections import Counter
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from elm_kernel import ELM
from sklearn import  metrics


def apply_feature_selection(X_train, X_test, y_train, feature_selection):
    if (feature_selection == "l1"):
        (X_train, X_test) = l1_feature_selection(X_train, X_test, y_train)
    elif feature_selection == "tree":
        (X_train, X_test) = tree_feature_selection(X_train, X_test, y_train)
    elif (feature_selection == "multi-level-f1"):
        (X_train_l1, X_test_l1) = l1_feature_selection(X_train, X_test, y_train)
        if X_train_l1.shape[1] == 0:
            return (None, None, None)
        (X_train, X_test) = tree_feature_selection(X_train_l1, X_test_l1, y_train)
    elif (feature_selection == "multi-level-f2"):
        (X_train_tree, X_test_tree) = tree_feature_selection(X_train, X_test, y_train)
        if X_train_tree.shape[1] == 0:
            return (None, None, None)
        (X_train, X_test) = l1_feature_selection(X_train_tree, X_test_tree, y_train)
    return (X_train, X_test)


def l1_feature_selection(train_features, test_features, y_train):
    clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_features, y_train)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(train_features)
    X_test_new = model.transform(test_features)
    print("X new after feature selection", X_new.shape)
    return (pd.DataFrame(X_new), pd.DataFrame(X_test_new))


def tree_feature_selection(train_features, test_features, y_train):
  clf = ExtraTreesClassifier(n_estimators=50)
  clf = clf.fit(train_features, y_train)
 # clf.feature_importances_
  model = SelectFromModel(clf, prefit=True)
  X_new = model.transform(train_features)
  X_new_test = model.transform(test_features)
  return (pd.DataFrame(X_new), pd.DataFrame(X_new_test))


def normalize_data(X_train, X_test, X_blind):
    scaler = StandardScaler()
    scaler = scaler.fit(X_train.values)
    X_train = pd.DataFrame(scaler.transform(X_train.values), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X_test.columns, index=X_test.index)
    print("X_blind shape is", X_blind.shape)
    X_blind = pd.DataFrame(scaler.transform(X_blind.values), columns=X_blind.columns, index=X_blind.index)

    return (X_train, X_test, X_blind)


def k_fold_cv_ensembling(X, y, feature_desc, is_normalize, feature_selection):
    id = 0
    avg_score = 0
    hard_pred = []
    soft_pred = []
    skf = StratifiedKFold(n_splits=3)
    for train_index, test_index in skf.split(X[0], y[0]):
        X_train, X_test = X[0].iloc[train_index], X[0].iloc[test_index]
        y_train, y_test = y[0].iloc[train_index], y[0].iloc[test_index]
        X_blind_temp = X[1]
        if is_normalize:
            (X_train, X_test, X_blind_temp) = normalize_data(X_train, X_test, X[1])
            if feature_selection:
                (X_train, X_test, X_blind_temp) = apply_feature_selection(X_train, X_test, y_train, feature_selection)

        if (X_train is None) or (X_train.shape[1] == 0):
            print("feature size is 0..")
            return (0, 0)
        clf = tune_ELM(X_train, y_train, X_test, y_test) if classifier == "ELM" else tune_on_devset(X_train, y_train,
                                                                                                    X_test, y_test)
        df = pd.DataFrame()
        df["test_index"] = test_index
        df["hard_pred"] = clf.predict(X_test)
        avg_score += evaluate(df["hard_pred"], y_test)
        ### BLIND SET PREDICTIONS FOR FOLD ID
        hard_pred.append(clf.predict(X_blind_temp))
        if classifier == "SVM":
            soft_pred.append(clf.predict_proba(X_blind_temp))
            df["soft_pred"] = clf.predict_proba(X_test).tolist()
        if is_write == True:
            df.to_csv(
                "/content/drive/My Drive/phd/ICMI-models/csv/No_FS/" + class_type + "_dev_preds_" + feature_desc + "_" + str(
                    id) + ".csv")
        print("Blind set score {} for fold {}", evaluate(hard_pred[-1], y[1]), str(id))
        id += 1
    print("AVG DEV SCORE IS:", avg_score / 3)
    if classifier == "SVM":
        avg_blind = write_preds(hard_pred, soft_pred, feature_desc)
    else:
        avg_blind = write_preds(hard_pred, None, feature_desc)

    return (avg_blind, avg_score / 3)


def majority(arr):
    # convert array into dictionary
    freqDict = Counter(arr)
    # traverse dictionary and check majority element
    size = len(arr)
    major_key = None
    major_val = 1
    for (key, val) in freqDict.items():
        if (val > (size / 2)):
            return key
        else:
            if val > major_val:
                major_key = key
                major_val = val
            elif val == major_val:
                major_key = None

    return major_key

def majority_voting_pred(predictions, weight_index=None):
    i = 0
    voted_preds = []
    len_pred = len(predictions[0])
    default_val = 2

    while i < len_pred:
        pred_arr = list(map(lambda x: x[i], predictions))
        mv = majority(pred_arr)
        if mv is None:
            if weight_index is not None:
                mv = predictions[weight_index][i]
            # Minority class is chosen if each classifier predicts a different label
            else:
                mv = default_val
        voted_preds.append(mv)
        i += 1
    return voted_preds

def print_confusion_matrix(y_blind, y_pred, feature_desc):
    print("(Majority voting) UAR score after CV, on test set with feature set (:", feature_desc,
    ")", evaluate(y_pred, y_blind))
    print('---accuracy---', '\n', metrics.accuracy_score(y_blind, y_pred), '\n')
    print('---confusion_matrix---', '\n', metrics.confusion_matrix(y_blind, y_pred), '\n')
    print('---classification_report---', '\n', metrics.classification_report(y_blind, y_pred), '\n')
    print ("---recall per class", recall_score(y_blind, y_pred,  average=None))


def write_preds(hard_pred, soft_pred, feature_desc):
    df = pd.DataFrame()
    mv_hard_preds = majority_voting_pred(hard_pred)
    print_confusion_matrix(y[1], mv_hard_preds, feature_desc)
    df["mv_hard_pred"] = mv_hard_preds
    df["hard_0"] = hard_pred[0]
    df["hard_1"] = hard_pred[1]
    df["hard_2"] = hard_pred[2]
    if soft_pred != None:
        df["soft_0"] = soft_pred[0].tolist()
        df["soft_1"] = soft_pred[1].tolist()
        df["soft_2"] = soft_pred[2].tolist()
        df["mv_soft_pred"] = np.mean(soft_pred, axis=0).tolist()

    if is_write == True:
        df.to_csv(
            "/content/drive/My Drive/phd/ICMI-models/csv/No_FS/" + class_type + "_blind_preds_" + feature_desc + "mv.csv")
    return evaluate(mv_hard_preds, y[1])

def tune_on_devset(X_train, y_train, X_devel, y_devel):
    uar_scores = []
    # score representations with SVM on different complexity levels
    complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 0.20, 0.25, 0.30, 0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.4, 1.5, 2, 10,
                    100]
    # C_range = log_C_range()
    # print("C range is :", C_range)
    # complexities = C_range
    kernels = ["sigmoid", "linear", "rbf"]
    gamma_val = [1e-1, 1, 1e1, 'scale']
    kernel_dict = {}
    complexity_dict = {}
    gamma_dict = {}
    best_train_uar = {}
    for k in kernels:
        for g in gamma_val:
            for c in complexities:
                clf = SVC(C=c, random_state=0, kernel=k, gamma=g, class_weight='balanced', probability=True).fit(
                    X_train, y_train)
                uar_scores.append(evaluate(clf.predict(X_devel), y_devel))
                best_train_uar[uar_scores[-1]] = evaluate(clf.predict(X_train), y_train)
                kernel_dict[uar_scores[-1]] = k
                complexity_dict[uar_scores[-1]] = c
                gamma_dict[uar_scores[-1]] = g
    UAR_dev = max(uar_scores)
    print("UAR dev score {} and UAR train {}", UAR_dev, best_train_uar[UAR_dev])
    print("best kernel and best comp pair: ", kernel_dict[UAR_dev], complexity_dict[UAR_dev], gamma_dict[UAR_dev])
    clf = SVC(C=complexity_dict[UAR_dev], random_state=0, kernel=kernel_dict[UAR_dev],
              gamma=gamma_dict[UAR_dev], class_weight='balanced', probability=True).fit(X_train, y_train)
    print("RECALL SCORE ON DEV FOR CHOSEN CLF:", recall_score(y_devel, clf.predict(X_devel), average=None))

    return clf


def tune_ELM(X_train, y_train, X_devel, y_devel):
    uar_scores = []
    # score representations with SVM on different complexity levels
    complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 0.20, 0.25, 0.30, 0.5, 0.8, 0.9, 1, 1.1, 1.2, 1.4, 1.5, 2, 10,
                    100]
    # C_range = log_C_range()
    # print("C range is :", C_range)
    # complexities = C_range
    complexity_dict = {}
    y_train = y_train.to_numpy()
    y_devel = y_devel.to_numpy()

    for c in complexities:
        clf = ELM(C=c, weighted=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_devel)
        uar_scores.append(evaluate(y_pred, y_devel))
        complexity_dict[uar_scores[-1]] = c
    UAR_dev = max(uar_scores)
    print("UAR dev score {} ", UAR_dev)
    print("best kernel and best comp pair: ", complexity_dict[UAR_dev])
    clf = ELM(C=complexity_dict[UAR_dev])
    clf.fit(X_train, y_train)

    return clf

def evaluate(y_pred, y):
    # Evaluation
    return (recall_score(y, y_pred, average='macro')* 100)


if __name__ == "__main__":
    classifier = "SVM"
    is_write = False
    class_type = "valence"

    ## Read Annotations  ##
    y = []
    # y[0] --- > TRAIN/DEV
    # y[1] --- > TEST (BLIND)
