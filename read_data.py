import pandas as pd
class Reader():
    '''Classe para carregamento e transformação inicial dos dados'''
    def read(self, path):
        dataframe = pd.read_fwf(path, header=None, engine='python')
        return dataframe

    def structure(self, dataframe):
        dataframe_split = pd.DataFrame(dataframe[0].str.split(',').tolist())
        dataframe_structured = dataframe_split[[0,1,2,3,4,5]]
        return dataframe_structured

    def format(self, dataframe):
        dataframe_nomed = dataframe.rename(columns={0:'individuo', 1:'atividade', 2:'timestamp', 3:'aceleracao_x', 4:'aceleracao_y', 5:'aceleracao_z'})
        dataframe_nomed['individuo'] = dataframe_nomed['individuo'].astype('str')
        dataframe_nomed['atividade'] = dataframe_nomed['atividade'].astype('str')
        dataframe_nomed['timestamp'] = dataframe_nomed['timestamp'].astype('int64')
        dataframe_nomed['aceleracao_x'] = dataframe_nomed['aceleracao_x'].astype('float64')
        dataframe_nomed['aceleracao_y'] = dataframe_nomed['aceleracao_y'].astype('float64')
        dataframe_nomed['aceleracao_z'] = dataframe_nomed['aceleracao_z'].str.replace(';', '')
        dataframe_nomed['aceleracao_z'] = dataframe_nomed['aceleracao_z'].fillna(0).replace('', 0)
        dataframe_nomed['aceleracao_z'] = dataframe_nomed['aceleracao_z'].astype('float64')
        return dataframe_nomed

    def derivate_fields(self, dataframe):
        dataframe['tempo_ns'] = pd.to_datetime(dataframe['timestamp'], unit='ns')
        classes = {'Jogging':0, 'Walking': 1, 'Upstairs':2, 'Downstairs':3, 'Sitting': 4, 'Standing':5}
        dataframe['class'] = dataframe['atividade'].map(classes)
        dataframe = dataframe.drop(columns=['atividade'])
        return dataframe





