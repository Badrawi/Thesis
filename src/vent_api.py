from csv_reader import CSVReader
class VentApi:
    NEGATIVE_EMOTION_CATEGORIES_IDS =['0af046e4-bf8d-4776-8d05-2e5128568330',
                             '2a1e8661-ff21-4dae-971e-587438ba35ea'
        ,'2f7d9ab1-c412-4734-9545-84b3cad01e59']
    POSITIVE_EMOTION_CATEGORIES_IDS = ['abdc8e31-96e9-40f9-8ca7-7a4687986074','79af7aa9-660e-49fc-9576-323b6a503179',
                                       '4fdc3a9c-48bc-4d8d-b01e-0d6e3097307c','79bcdf8a-14fa-4b54-a21c-022f69395c37']
    def __init__(self):
        self.emotion_data = CSVReader.dataframe_from_file("VentDataset/emotions.csv", ['id', 'emotion_category_id'])
       # self.emotion_data = self.emotion_data[self.emotion_data.enabled == 'TRUE']
        self.vent_data = CSVReader.dataframe_from_file("VentDataset/vents.csv",['emotion_id','text'])
    def getPositiveEmotions(self):
        data = self.emotion_data
        return data[data.emotion_category_id.isin(self.POSITIVE_EMOTION_CATEGORIES_IDS)]
    def getNegativeEmotions(self):
        data = self.emotion_data
        return data[data.emotion_category_id.isin(self.NEGATIVE_EMOTION_CATEGORIES_IDS)]

    def getPositiveVents(self):
        data = self.vent_data
        emotions = self.getPositiveEmotions()
        data = data[data.emotion_id.isin(emotions.id)]
        return data.text
    def getNegativeVents(self):
        data = self.vent_data
        emotions = self.getNegativeEmotions()
        data = data[data.emotion_id.isin(emotions.id)]
        return data.text