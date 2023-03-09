import pandas as pd
class Filter():
    '''Classe para aplicação de filtros na base de treinamento'''
    def group_interval(self, dataframe):
        dataframe_3_sec = self.filter_by_3_sec(dataframe)
        dataframe_5_sec = self.filter_by_5_sec(dataframe)
        dataframe_interval =  pd.concat([dataframe_3_sec, dataframe_5_sec])
        return dataframe_interval

    def clean_dataset(self, dataframe):
        dataframe = dataframe.drop_duplicates(subset=['timestamp'])
        dataframe['tempo_ns'] = pd.to_datetime(dataframe['timestamp'], unit='ns')
        dataframe.set_index('tempo_ns', inplace=True)
        dataframe_no_null = dataframe.dropna(how='all')
        return dataframe_no_null

    def filter_by_3_sec(self, dataframe):
        interval_by_3s = dataframe.groupby(pd.Grouper(key='tempo_ns', freq='3S'))
        first_group_element_3s = interval_by_3s.first()
        first_group_element_3s.reset_index(drop=True)
        last_group_element_3s = interval_by_3s.last()
        last_group_element_3s.reset_index(drop=True)
        dataframe_interval =  pd.concat([first_group_element_3s, last_group_element_3s])
        return dataframe_interval

    def filter_by_5_sec(self, dataframe):
        interval_by_5s = dataframe.groupby(pd.Grouper(key='tempo_ns', freq='5S'))
        first_group_element_5s = interval_by_5s.first()
        first_group_element_5s.reset_index(drop=True)
        return first_group_element_5s
