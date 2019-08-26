import pandas as pd
import numpy as np
from text_preprocessing import TextPreprocessing
import xlrd
class CSVReader:
    @staticmethod
    def text_sentiment_map(file,cols):
        fpath = "../data/"+file
        data = pd.read_csv(fpath,sep=',',
             quotechar="'",usecols=cols)
        data =data.dropna()
        return data

