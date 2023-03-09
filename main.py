from read_data import Reader
from filter_dataset import Filter
from pre_processing import PreProcessing
from process_model import Model
from evaluate_model import EvaluateModel
from generate_matrix import GenerateMatrix

if __name__ == '__main__':

    '''Aplicação principal para geração do modelo'''

    #Seleção do diretório do arquivo a ser processado
    path = input('Por favor, informe o endereço(diretório) do arquivo para processamento: ')

    #Seleção do modelo a ser utilizado
    modelos = ['rf', 'xgb']
    acuracia = {}
    relatorios = {}
    matrizes = {}

    for model in modelos:
        #Realiza carregamento e estruturação inicial do dataset
        reader = Reader()
        dataframe = reader.derivate_fields(reader.format(reader.structure(reader.read(path))))

        #Aplica um filtro para considerar um intervalo dos dados de treinamento entre 3 e 5 segundos e termina de estruturar o dataset
        filter = Filter()
        dataframe_filtered = filter.clean_dataset(filter.group_interval(dataframe))

        #Realiza pre-processamento, particionando o dataset em treino e teste
        particoes = PreProcessing().partitions_dataset(dataframe_filtered, test_size=0.2)

        #Aplica a seleção do modelo e realiza o processamento
        modelo = Model.instanciaModelo(model)
        predicao = Model.compute_models(modelo, particoes)

        #Avalia o modelo, obtém métrica de acurácia, relatório de classificação
        avaliacao = EvaluateModel()
        acuracia[model] = avaliacao.getAcuracia(particoes['y_test'], predicao)
        relatorios[model] = avaliacao.getClassificationReport(particoes['y_test'], predicao)

        print('---------------------------Acurácia-----------------------------------')
        print(f'Acurácia = {model}: {acuracia[model]}')
        print('---------------------------Relatório----------------------------------')
        print(relatorios[model])

        #Gera matrix de confusão do modelo
        labels = dataframe_filtered['class'].unique().tolist()
        matrizes[modelo] = GenerateMatrix(particoes, predicao, labels).plot(model)


