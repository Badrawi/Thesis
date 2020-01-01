import pandas as pd
import numpy as np
from text_preprocessing import TextPreprocessing
import xlrd
class CSVReader:
    @staticmethod
    def dataframe_from_file():
        fpath = "glue_data/MRPC/train.tsv"
        data = pd.DataFrame.read_csv(fpath,sep='\t')
        data =data.dropna()
        return data

