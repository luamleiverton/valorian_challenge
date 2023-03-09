from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score


class EvaluateModel():
    '''Métodos de avaliação dos modelos'''
    def getAcuracia(self, y_real, y_pred):
        return balanced_accuracy_score(y_real, y_pred)

    def getClassificationReport(self, y_real, y_pred):
        return classification_report(y_real, y_pred)