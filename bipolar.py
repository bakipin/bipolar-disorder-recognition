#%%
import os
import numpy as np
import csv
from sklearn.metrics import confusion_matrix, classification_report
from elm_kernel import ELM
from permutation_importance import PermutationImportance
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
import seaborn as sn
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#%%

#save mania level labels into a dict
def save_labels():
    train_dict = {}
    dev_dict = {}
    with open('labels/labels_metadata_AVEC_traindevel.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if row[6] == 'train':
                    train_dict[row[0]] = int(row[5])
                elif row[6] == 'dev':
                    dev_dict[row[0]] = int(row[5])

    test_dict = {}
    with open('labels/test_labels.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            test_dict[row[0]] = int(row[1])
    return train_dict, dev_dict, test_dict

#read summary features from csv files
def read_summary_features(feature_set):
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []

    train_dict, dev_dict, test_dict = save_labels()
    for (dirpath, dirnames, filenames) in os.walk("features/clip_"+feature_set+"_summary/"):
        for filename in filenames:
            if filename.endswith(".csv"):
                with open(dirpath + filename) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        if filename[:5] == 'train':
                            x_train.append([float(i) for i in row])
                            y_train.append(train_dict[filename[:-4]]-1)
                        elif filename[:3] == 'dev':
                            x_dev.append([float(i) for i in row])
                            y_dev.append(dev_dict[filename[:-4]]-1)
                        else:
                            x_test.append([float(i) for i in row])
                            y_test.append(test_dict[filename[:-4]]-1)

    x = np.concatenate((x_train,x_dev), axis=0)
    y = np.concatenate((y_train,y_dev), axis=0)
    return x_train, y_train, x_dev, y_dev, x_test, y_test, x, y

def read_IS10_features():
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []
    for filename in os.listdir('features/IS10/'):
        if filename.endswith('.csv'):
            with open('features/IS10/'+filename) as file:
                csv_reader = csv.reader(file, delimiter=',')
                for row in csv_reader:
                    if filename.__contains__('train'):
                        x_train.append([float(i) for i in row[:-1]])
                        y_train.append(int(row[-1])-1)
                    elif filename.__contains__('dev'):
                        x_dev.append([float(i) for i in row[:-1]])
                        y_dev.append(int(row[-1])-1)
                    else:
                        x_test.append([float(i) for i in row[:-1]])
                        y_test.append(int(row[-1])-1)
    x_train = np.array(x_train)
    x_dev = np.array(x_dev)
    x_test = np.array(x_test)

    x = np.concatenate((x_train,x_dev), axis=0)
    y = np.concatenate((y_train,y_dev), axis=0)
    return x_train, y_train, x_dev, y_dev, x_test, y_test, x, y

#reak task features from csv files
#tasks is a list containing task names, for example tasks = ['T123', 'T45']
def read_task_features(feature_set, tasks):
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    x_test = []
    y_test = []

    dev_files = []
    test_files = []
    train_dict, dev_dict, test_dict = save_labels()
    for (dirpath, dirnames, filenames) in os.walk("features/task_"+feature_set+"_summary/"):
        for filename in filenames:
            if filename.endswith(".csv"):
                with open(dirpath + filename) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        for task in tasks:
                            if filename[:5] == 'train' and (task in filename):
                                x_train.append([float(i) for i in row])
                                y_train.append(train_dict[filename[:9]]-1)
                            elif filename[:3] == 'dev' and (task in filename):
                                x_dev.append([float(i) for i in row])
                                y_dev.append(dev_dict[filename[:7]]-1)
                                dev_files.append(filename)
                            elif filename[:4] == 'test' and (task in filename):
                                x_test.append([float(i) for i in row])
                                y_test.append(test_dict[filename[:8]]-1)
                                test_files.append(filename)

    x = np.concatenate((x_train,x_dev), axis=0)
    y = np.concatenate((y_train,y_dev), axis=0)
    return x_train, y_train, x_dev, y_dev, x_test, y_test, x, y, dev_files, test_files, dev_dict, test_dict

#used for combining results of the tasks on each clip
def combine_task_results(pred, files, dict):
    scores = {}
    for i in range(len(pred)):
        if scores.get(files[i][:7]) is None:
            scores[files[i][:7]] = []
        scores[files[i][:7]].append(list(pred[i]))
    preds = []
    y = []
    check_list = []
    for key, val in scores.items():
        preds.append(np.array(val).sum(axis=0) / len(val))
        y.append(dict[key] - 1)
        check_list.append(key)
    for key, val in dict.items():
        if key not in check_list:
            y.append(dict[key]-1)
    return preds, y

#scaling, first Z normalization, then l2 normalization
def scale_features(x_train, x_dev):
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_dev = scaler.transform(x_dev)
    x_train = [clip/np.linalg.norm(clip) for clip in x_train]
    x_dev = [clip/np.linalg.norm(clip) for clip in x_dev]
    return np.array(x_train), np.array(x_dev)

#apply PCA to the training set, fit on development set
def apply_PCA(x_train, x_dev, variance):
    pca = PCA(variance)
    x_train = pca.fit_transform(x_train)
    x_dev = pca.transform(x_dev)
    return x_train, x_dev

#plot onfusion matrix
def plot_confusion_matrix(cf):
    df_cm = pd.DataFrame(cf, index=['Hypomania', 'Mania', 'Remission'],
                         columns=['Hypomania', 'Mania', 'Remission'])

    plt.figure(figsize=(5, 4))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.show()

#single level elm model, returns the model and labels list
def get_elm_results(x_train, y_train, x_dev, y_dev, C, weighted):
    elm = ELM(C=C, weighted=weighted)
    elm.fit(x_train, y_train)
    y_pred1 = elm.predict(x_dev)
    y_pred = [np.argmax(pred) for pred in y_pred1]
    cf = confusion_matrix(y_dev, y_pred)
    plot_confusion_matrix(cf)
    target_names = ['remission','hypomania','mania']
    print(classification_report(y_dev, y_pred, target_names=target_names, digits=4))
    return elm, y_pred

#combination of weighted and unweighted elm
def get_fusion_elm_results(x_train, y_train, x_dev, y_dev, C_weighted, C_unweighted):
    weighted_elm = ELM(C=C_weighted, weighted=True)
    unweighted_elm = ELM(C=C_unweighted, weighted=False)

    weighted_elm.fit(x_train, y_train)
    unweighted_elm.fit(x_train, y_train)
    y_pred_weighted = weighted_elm.predict(x_dev)
    y_pred_unweighted = unweighted_elm.predict(x_dev)

    scores = []
    for alpha in np.linspace(0,1,21):
        y_pred_mean = (alpha*y_pred_unweighted) + ((1-alpha)*y_pred_weighted)
        pred = [np.argmax(mean) for mean in y_pred_mean]
        scores.append(recall_score(y_dev, pred, average='macro'))

    alpha = np.linspace(0,1,21)[np.argmax(scores)]

    y_pred_mean = (alpha * y_pred_unweighted) + ((1 - alpha) * y_pred_weighted)
    pred = [np.argmax(mean) for mean in y_pred_mean]
    cf = confusion_matrix(y_dev, pred)
    #plot_confusion_matrix(cf)

    target_names = ['remission','hypomania','mania']
    #print(classification_report(y_dev, pred, target_names=target_names, digits=4))
    return alpha, scores, weighted_elm, unweighted_elm, y_pred_mean

# perform permutation importance
def get_importances(x,y, model, group, labels):
    importances = PermutationImportance(model).fit(x, y, group=group)
    # plot feature importance
    fig, ax = plt.subplots()
    plt.bar(range(len(importances)), list(importances.values()), align='center')
    if len(labels) > 0:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')
    ax.axhline(0, color='black')
    plt.show()

#%%
#stacking
def meta_learning(x_train, y_train, x_dev, y_dev, C_weighted, C_unweighted):
    weighted_elm = ELM(C=C_weighted, weighted=True)
    unweighted_elm = ELM(C=C_unweighted, weighted=False)
    knn = KNeighborsClassifier(n_neighbors=3)
    rfc = RandomForestClassifier(random_state=1, n_estimators=100)

    #models = [unweighted_elm, weighted_elm, weighted_elm2, unweighted_elm2]
    models = [rfc, weighted_elm, unweighted_elm, knn]
    holdout = np.zeros((60,3))
    dev_num = int(len(x_train)/len(models))
    x_train, x_dev = scale_features(x_train, x_dev)
    x_train, x_dev = apply_PCA(x_train, x_dev, 0.99)
    print(x_train.shape)
    for i in range(len(models)):
        x_train_temp = np.concatenate((x_train[:i*dev_num],x_train[i*dev_num+dev_num:]), axis=0)
        y_train_temp = y_train[:i*dev_num] + y_train[i*dev_num+dev_num:]
        x_dev_temp = np.array(x_train[i*dev_num:i*dev_num+dev_num])
        y_dev_temp = y_train[i*dev_num:i*dev_num+dev_num]

        models[i].fit(x_train_temp, y_train_temp)
        preds = models[i].predict_proba((x_dev_temp))
        '''
        if i==0 or i==1: preds = models[i].predict_proba(x_dev_temp)
        else: preds = models[i].predict_proba(x_dev_temp)
 
        y_preds = []
        for l in preds:
            l = np.exp(l - np.max(l))
            l = l/np.sum(l)
            y_preds.append(l)
        '''
        y_preds = preds
        if i==0: first_level = y_preds
        else: first_level = np.concatenate((first_level, y_preds), axis=0)

        holdout_preds = models[i].predict_proba(x_dev)
        y_preds = holdout_preds
        holdout += y_preds
        y_preds = [np.argmax(mean) for mean in y_preds]
        print(recall_score(y_dev, y_preds, average='macro'))
    holdout /= 4

    svm_model = SVC(kernel='linear',gamma=1, C=1000, max_iter=-1).fit(first_level, y_train)
    y_pred = svm_model.predict(holdout)

    print(recall_score(y_dev, y_pred, average='macro'))



#EXPERIMENTS
#%%
#meta learning
x_train, y_train, x_dev, y_dev, x_test, y_test, x, y = read_summary_features('eGEMAPS')
meta_learning(x_train,y_train,x_dev,y_dev,1000,10)

#%%
#cross validation meta learning
x_train, y_train, x_dev, y_dev, x_test, y_test, x, y = read_summary_features('eGEMAPS')
k=4
dev_num = int(len(x)/k)
for n in range(k):
    x_train = np.concatenate((x[:n*dev_num],x[n*dev_num+dev_num:]), axis=0)
    y_train = np.concatenate((y[:n*dev_num],y[n*dev_num+dev_num:]), axis=0)
    x_dev = x[n*dev_num:n*dev_num+dev_num]
    y_dev = y[n*dev_num:n*dev_num+dev_num]
    meta_learning(x_train, y_train, x_dev, y_dev, 1000, 10)

#%%
#cross validation fusion elm
x_train, y_train, x_dev, y_dev, x_test, y_test, x, y = read_summary_features('eGEMAPS')
k=4
dev_num = int(len(x)/k)
for n in range(k):
    x_train = np.concatenate((x[:n*dev_num],x[n*dev_num+dev_num:]), axis=0)
    y_train = np.concatenate((y[:n*dev_num],y[n*dev_num+dev_num:]), axis=0)
    x_dev = x[n*dev_num:n*dev_num+dev_num]
    y_dev = y[n*dev_num:n*dev_num+dev_num]
    x_train, x_dev = scale_features(x_train, x_dev)
    alpha, scores, weighted_elm, unweighted_elm, y_pred_mean = get_fusion_elm_results(x_train, y_train, x_dev, y_dev, 100, 10)
    print(np.max(scores))
    print(alpha)

labels = ['loudness', 'alpha ratio', 'hammerberg index', 'spectral 0-500 Hz', 'spectral 500-1500 Hz'
    , 'spectral flux', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'pitch', 'jitter', 'shimmer'
    , 'harmonic-to-noise ratio', 'harmonic difference H1-H2', 'harmonic difference H1-A3'
    , 'F1 freq', 'F1 bandwidth', 'F1 amplitude', 'F2 freq', 'F2 amplitude', 'F3 freq', 'F3 amplitude']
#labels = ['mean', 'stddev', 'curv coeff', 'slope', 'offset', 'min', 'rel. min', 'max', 'rel. max', 'range']
#get_importances(x_dev, y_dev, weighted_elm, 'feature', labels)

#%%
#experiments on tasks
tasks = ['T7']
x_train, y_train, x_dev, y_dev, x_test, y_test, x, y, dev_files, test_files, dev_dict, test_dict = read_task_features('eGEMAPS', tasks)
x_train, x_dev = scale_features(x_train, x_dev)
x_train, x_dev = apply_PCA(x_train, x_dev, 0.99)
alpha, scores, weighted_elm, unweighted_elm, pred = get_fusion_elm_results(x_train, y_train, x_dev, y_dev, 100, 10)
preds, y_dev = combine_task_results(pred, dev_files, dev_dict)

pred = [np.argmax(mean) for mean in preds]
if len(pred)<60:
    pred.extend([1]*(60-len(pred)))
cf = confusion_matrix(y_dev, pred)
plot_confusion_matrix(cf)

target_names = ['remission', 'hypomania', 'mania']
print(classification_report(y_dev, pred, target_names=target_names, digits=4))