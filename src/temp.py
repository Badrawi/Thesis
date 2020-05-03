import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,Callback,TensorBoard
from tensorflow.keras.models import load_model,Sequential,Model
from models import Models
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from text_preprocessing import TextPreprocessing
import json
import os
from csv_reader import CSVReader
from datetime import datetime
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report
textPreProcessing  = TextPreprocessing()
if __name__ == "__main__":
    embeddings = np.load('text_embedding.npy', allow_pickle=True)
    texts = []
    sentiments = []
    good = CSVReader.dataframe_from_txt("WordNetAffectEmotionLists/joy.txt")
    good = good.texts
    texts = texts = np.append(texts,np.array(good))
    sentiments = np.append(sentiments,[1]*len(good))
    surprise = CSVReader.dataframe_from_txt("WordNetAffectEmotionLists/surprise.txt")
    surprise = surprise.texts
    texts = np.append(texts,np.array(surprise))
    sentiments = np.append(sentiments,[2]*len(surprise))
    sad = CSVReader.dataframe_from_txt("WordNetAffectEmotionLists/sadness.txt")
    sad = sad.texts
    texts = np.append(texts,np.array(sad))
    sentiments = np.append(sentiments,[-1]*len(sad))
    fear = CSVReader.dataframe_from_txt("WordNetAffectEmotionLists/fear.txt")
    fear = fear.texts
    texts = np.append(texts,np.array(fear))
    sentiments = np.append(sentiments,[-2]*len(fear))
    anger = CSVReader.dataframe_from_txt("WordNetAffectEmotionLists/anger.txt")
    anger = anger.texts
    texts = np.append(texts,np.array(anger))
    sentiments = np.append(sentiments,[-2]*len(anger))
    texts = np.append(texts,["hungry","starving","want to eat"])
    sentiments = np.append(sentiments,[-2]*3)
    texts = np.append(texts,["awake at night"])
    sentiments = np.append(sentiments,[-1])
    texts = np.append(texts,["full"])
    sentiments = np.append(sentiments,[1])
    texts = np.append(texts,["not hungry","at all","excited","awesome","cool"])
    sentiments = np.append(sentiments,[2]*5)
    texts = np.append(texts,["ok"])
    sentiments = np.append(sentiments,[0])
    airline_data = CSVReader.dataframe_from_file("Tweets.csv",['airline_sentiment','text'])
    airline_text = np.array(airline_data.text)
    airline_sentiment = np.array(airline_data.airline_sentiment)
    count = 0
    tokenizer = Tokenizer(num_words=300000)
    tokenizer.fit_on_texts(texts)
    vocab = np.array([w for w,_ in tokenizer.word_index.items()])
    for i in range(len(airline_sentiment)):
        if(count > 1000):
            break
        if (airline_sentiment[i] == "neutral"):
            airline_text[i] = textPreProcessing.remove_special_characters(airline_text[i], True)
            airline_text[i] = textPreProcessing.remove_accented_chars(airline_text[i])
            airline_text[i] = textPreProcessing.remove_whiteList(airline_text[i])
            text = airline_text[i]
            words = textPreProcessing.tokenize(text)
            words = [w for w in words if w not in vocab]
            if(len(words) > 0):
                count += 1
                vocab = np.append(vocab,[words[0]])
    texts = np.append(texts,vocab)
    sentiments = np.append(sentiments,[0]*len(vocab))
    categorical_sentiments = to_categorical(sentiments,num_classes=5)
    tokenizer = Tokenizer(num_words=300000)
    tokenizer.fit_on_texts(texts)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    filepath =  "return.h5"
    models = Models()
    models.buil_pre_model(embeddings)
    model = models.model
    if os.path.isfile(filepath):
        model = load_model(filepath)

    model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                            metrics=['acc'])
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,tensorboard_callback]
    
    model.fit(pad_sequences(tokenizer.texts_to_sequences(texts),maxlen=150),
                      categorical_sentiments,
                       batch_size=32,epochs=50,validation_data=(pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=150)
                                                                 ,Y_test),callbacks=callbacks_list,shuffle=True)

    result = model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences([
                                                                               " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
                                                                                     " I sleep hungry and It gets harder everyday",
                                                                                     "everything is great, i have lost some weight",
                                                                                     "awesome, really cool",
                                                                                     "should I play cards",
                                                                                     "I am full and inshape",
                                                                                     "is it okay to be that hungry at night?"])
                                                 ,maxlen=150))
    print("result: ", np.argmax(result,axis=-1),"\n")
