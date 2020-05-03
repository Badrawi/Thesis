import pandas as pd
import numpy as np
import xlrd
class CSVReader:
    @staticmethod
    def dataframe_from_file(file,cols):
        fpath = "../data/"+file
        data = pd.read_csv(fpath,sep=',',
             quotechar="'",usecols=cols)
        data =data.dropna()
        return data
    @staticmethod
    def dataframe_from_txt(file):
        fpath = "../data/"+file
        data = pd.read_csv(fpath,sep=',',usecols=["num","texts"])
        # data.columns = ["num","texts"]
        return data