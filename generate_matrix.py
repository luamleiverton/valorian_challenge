from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

class GenerateMatrix():
    '''Classe de geração da matriz de confusão'''
    def __init__(self, particoes, pred, labels):
        self.y_real = particoes['y_test']
        self.pred = pred
        self.cm = confusion_matrix(particoes['y_test'], pred, labels=labels)
        self.annot = None

    def get_anotation(self):
        cm_sum = np.sum(self.cm, axis=1, keepdims=True)
        cm_perc = self.cm / cm_sum.astype(float) * 100
        self.annot = np.empty_like(self.cm).astype(str)
        nrows, ncols = self.cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = self.cm[i, j]
                p = cm_perc[i, j]
                if c == 0:
                    self.annot[i, j] = ''
                else:
                    s = cm_sum[i]
                    self.annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        return self.annot

    def plot(self, modelo):
        plt.switch_backend('AGG')
        cm_rf_df = pd.DataFrame(self.cm, index=np.unique(self.y_real), columns=np.unique(self.pred))
        cm_rf_df.index.name = 'Actual'
        cm_rf_df.columns.name = 'Predicted'
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        sns.heatmap(self.cm, annot=self.get_anotation(), fmt='', cmap='Greens', ax=ax)
        plt.title('Confusion matrix')
        plt.savefig(f'confusionmatrix_{modelo}.png', format='png')


