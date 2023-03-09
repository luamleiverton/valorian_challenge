import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFRegressor, XGBRegressor, XGBModel, XGBClassifier
import time

class Model():
    '''Processamento do modelo'''
    @staticmethod
    def instanciaModelo(modelo):
        if modelo == 'rf':
            return RandomForestClassifier(n_estimators=50, random_state=123)
        elif modelo == 'xgb':
            return XGBClassifier()
        else:
            raise Exception('Modelo não incorporado')

    @staticmethod
    def compute_models(modelo, particoes):
        inicio = time.time()
        model_fitted = modelo.fit(particoes['x_train'], particoes['y_train'])
        # Fit the model
        print('Model fitted:', model_fitted)
        # Make predictions
        y_pred = model_fitted.predict(particoes['x_test'])
        termino = time.time()
        print(f'tempo: {termino - inicio}')
        return y_pred








