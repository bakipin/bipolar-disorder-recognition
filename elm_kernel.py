import numpy as np
import sklearn.metrics
import sys
np.set_printoptions(threshold=sys.maxsize)

class ELM():
    def __init__(self, C=1, weighted=True, kernel='rbf'):
        super(self.__class__, self).__init__()

        self.x_train = []
        self.C = C
        self.weighted = weighted
        self.beta = []
        self.kernel = kernel

    def fit(self, x_train, y_train):
        self.x_train = x_train
        class_num = max(y_train) + 1
        n = len(x_train)
        y_onehot = np.eye(class_num)[y_train]

        if self.kernel == 'rbf':
            kernel_func = sklearn.metrics.pairwise.rbf_kernel(x_train)

        if self.weighted:
            W = np.identity((n))
            hist = np.zeros(class_num)
            for label in y_train:
                hist[label] += 1
            hist = 1/hist
            for i in range(len(y_train)):
                W[i,i] = hist[y_train[i]]
            beta = np.matmul(np.linalg.inv(np.matmul(W,kernel_func) + np.identity(n) / self.C), np.matmul(W,y_onehot))
        else:
            beta = np.matmul(np.linalg.inv(kernel_func + np.identity(n) / self.C), y_onehot)
        self.beta = beta
        return beta


    def predict(self, x_test):
        if self.kernel == 'rbf':
            kernel_func = sklearn.metrics.pairwise.rbf_kernel(x_test, self.x_train)
        pred = np.matmul(kernel_func, self.beta)
        '''scaled = []
        for l in pred:
            if np.sum(l) == 0: scaled.append(np.array([0.33,0.33,0.33]))
            else: scaled.append(l/np.sum(l))
        return [np.argmax(pred) for pred in scaled]'''
        return pred

    def predict_proba(self, x_test):
        if self.kernel == 'rbf':
            kernel_func = sklearn.metrics.pairwise.rbf_kernel(x_test, self.x_train)
        pred = np.matmul(kernel_func, self.beta)
        scaled = []
        '''for l in pred:
            if np.sum(l) == 0: scaled.append(np.array([0.33,0.33,0.33]))
            else: scaled.append(l/np.sum(l))
        return np.array(scaled)'''
        return pred