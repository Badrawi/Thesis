from csv_reader import CSVReader
import numpy as np
from text_preprocessing import TextPreprocessing
class VentApi:
    EMOTION_GOOD_ID =['96ccdd2e-152e-4e0d-a6da-c49d1c68b10c','0893f043-1382-4d65-9378-2fb89ada1091'
                             'e0babb52-73cd-4e31-b83c-357983cc7b46','d84f9579-ba96-4818-93a5-7ff12f504098'
        ,'f6b390e3-a891-40a3-9d5b-c7ca2c5b39d6','846c2090-db10-4b5e-a748-7679c747de01','c88cc72e-d203-4f3d-a829-ed5ac7ed4a2c']
    EMOTION_ENERGIZED_ID  = ['4fc35470-9e4a-487c-9080-dac6ac31002c','91a437fd-8c2e-4fe4-a6fe-bf972cbcaf9e',
                                       '561926ab-ef1d-4eb8-98d8-5eeba974df2c','b1ce7817-4e73-413d-98f8-9f38be0e2970',
                             '7b98865e-4fa5-4fec-b2b0-d82263686124']
    EMOTION_BAD_ID = ['3180a95c-c03d-4a36-b78c-26d54d928049','cc6b6ea7-7b2c-4f04-85cb-5c0b4a8cd351',
                      '2fc407ef-82c1-425c-80a7-2aa497610f2e','c57f36ec-dbc1-455f-ae97-7ef124431ea2',
                      'c0dd0bcf-81c3-48d8-b75b-e150c24162bc','f59fced6-6d10-4bb0-a0f2-5c0fec7ed687','fcd17f8b-5010-40d2-ab7a-31d699568b95']
    EMOTION_STRUGGLE_ID = ['fe1ac197-3294-493f-ba9d-04c6bfbea10c','730d4894-def4-4b34-ae23-ea4fc91d8104',
                           '0dbaa1cc-8ceb-4798-94ea-55587b5f1ef9','e2166c53-90b0-4d7c-a87f-b97b4793c30b',
                           'b01b8e6c-3fd5-4ca2-aa8f-3fef176ac4d1','42935e4e-2c07-4bda-8377-d74d4cf817d7','9632cb01-b6ab-4190-9aef-239afacfc742']
    EMOTION_NEUTRAL_ID = ['dc864085-fc61-4b62-b80e-54e0f0580ba8','d8178412-aec1-4b5a-bb28-c73e0583f3dc']
    def __init__(self):
        #self.emotion_data = CSVReader.dataframe_from_file("VentDataset/emotions.csv", ['id', 'emotion_category_id'])
       # self.emotion_data = self.emotion_data[self.emotion_data.enabled == 'TRUE']
        self.vent_data = CSVReader.dataframe_from_file("VentDataset/vents.csv",['emotion_id','text'])
        self.textPreProcessing = TextPreprocessing()

    def getVents(self,array):
        vent_data =  self.vent_data[self.vent_data.emotion_id.isin(array)]
        texts = np.array(vent_data.text)
        for i in range(len(texts)):
          texts[i] = self.textPreProcessing.remove_special_characters(texts[i], True)
          texts[i] = self.textPreProcessing.remove_accented_chars(texts[i])
          texts[i] = self.textPreProcessing.remove_whiteList(texts[i])
        return texts