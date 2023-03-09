import numpy as np
from sklearn.model_selection import train_test_split

class PreProcessing():
    '''Método de pré-processamento dos dados'''
    def __init__(self):
        self.seed = 10

    def partitions_dataset(self, dataframe, test_size):
        particoes = {}
        X = np.array(dataframe.drop('class', axis=1))
        y = dataframe['class']
        particoes['x_train'], particoes['x_test'], particoes['y_train'], particoes['y_test'] = train_test_split(X, y, test_size=test_size, random_state = self.seed, stratify=y)
        return particoes


