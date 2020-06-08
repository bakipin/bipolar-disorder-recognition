import numpy as np
from sklearn.metrics import recall_score
import random

class PermutationImportance():
    def __init__(self, model):
        super(self.__class__, self).__init__()

        self.model = model

    def fit(self, x, y, group = 'individual'):
        y_pred1 = self.model.predict(x)
        y_pred = [np.argmax(pred) for pred in y_pred1]
        orig_acc = recall_score(y, y_pred, average='macro')

        if group == 'feature':
            x_copy = np.array(x)
            importances = {}
            feat_num = len(x[0])/10
            for i in range(int(feat_num)):
                x_trans = np.array(np.transpose(x_copy))
                for j in range(10):
                    random.Random(0).shuffle(x_trans[int(i * 10 + j)])
                y_pred1 = self.model.predict(np.transpose(x_trans))
                y_pred = [np.argmax(pred) for pred in y_pred1]
                new_acc = recall_score(y, y_pred, average='macro')

                importances[i] = orig_acc - new_acc

        if group == 'function':
            x_copy = np.array(x)
            importances = {}
            feat_num = len(x[0])/10
            for i in range(10):
                x_trans = np.array(np.transpose(x_copy))
                for j in range(int(feat_num)):
                    random.Random(0).shuffle(x_trans[int(i*feat_num+j)])
                y_pred1 = self.model.predict(np.transpose(x_trans))
                y_pred = [np.argmax(pred) for pred in y_pred1]
                new_acc = recall_score(y, y_pred, average='macro')

                importances[i] = orig_acc - new_acc

        if group == 'individual':
            x_copy = np.array(x)
            importances = {}
            for i in range(len(x[0])):
                x_trans = np.array(np.transpose(x_copy))
                random.Random(0).shuffle(x_trans[i])
                y_pred1 = self.model.predict(np.transpose(x_trans))
                y_pred = [np.argmax(pred) for pred in y_pred1]
                new_acc = recall_score(y, y_pred, average='macro')

                importances[i] = orig_acc - new_acc

        return importances

