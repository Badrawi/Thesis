import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,Callback,TensorBoard
from tensorflow.keras.models import load_model,Sequential,Model
from models import Models
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import json
import os
from csv_reader import CSVReader
from datetime import datetime
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report

def train():
    embeddings = np.load('text_embedding.npy', allow_pickle=True)
    sentiments = np.load('sentiments.npy', allow_pickle=True)
    texts = np.load('texts.npy', allow_pickle=True)
    all_texts = np.load('text_cache.npy', allow_pickle=True)
    categorical_sentiments = to_categorical(sentiments,num_classes=5)
    tokenizer = Tokenizer(num_words=300000, oov_token=None)
    tokenizer.fit_on_texts(all_texts)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)
    np.save("text_train.npy",X_train)
    np.save("sentiment_train.npy",Y_train)
    models = Models()
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    models = []
    bgruModel =  "ensemble_bgru.h5"
    models.buil_pre_model(embeddings)
    model = models.model
    if os.path.isfile(filepath):
        model = load_model(filepath)

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,tensorboard_callback]

    model.fit(pad_sequences(tokenizer.texts_to_sequences(X_train[:500000]),maxlen=75),
                      Y_train[:500000],
                       batch_size=512,epochs=50,validation_data=(pad_sequences(tokenizer.texts_to_sequences(X_test[:5000]),maxlen=75)
                                                                 ,Y_test[:5000]),callbacks=callbacks_list,shuffle=True)

    result = model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences([
                                                                               " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
                                                                                     " I sleep hungry and It gets harder everyday",
                                                                                     "everything is great, i have lost some weight",
                                                                                     "awesome, really cool",
                                                                                     "should I play cards",
                                                                                     "I am full and inshape",
                                                                                     "is it okay to be that hungry at night?"])
                                                 ,maxlen=75))
    print("result: ", np.argmax(result,axis=-1),"\n")



if __name__ == "__main__":
    embeddings = np.load('text_embedding.npy', allow_pickle=True)
    sentiments = np.load('sentiments.npy', allow_pickle=True)
    texts = np.load('texts.npy', allow_pickle=True)
    all_texts = np.load('text_cache.npy', allow_pickle=True)
    tokenizer = Tokenizer(num_words=300000, oov_token=None)
    tokenizer.fit_on_texts(all_texts)
    _, X_test, _, Y_test = train_test_split(texts, sentiments, test_size=0.01)
    positive = []
    negative = []
    neutral = []
    motivated = []
    struggling =[]
    for i in range(len(Y_test)):
        if(Y_test[i] == 1):
            positive = np.append(positive,X_test[i])
        if(Y_test[i] == 2):
            motivated = np.append(motivated,X_test[i])
        if(Y_test[i] == 0):
            neutral = np.append(neutral,X_test[i])
        if(Y_test[i] == -1):
            negative = np.append(negative,X_test[i])
        if(Y_test[i] == -2):
            struggling = np.append(struggling,X_test[i])

    airline_data = CSVReader.dataframe_from_file("Tweets.csv",['airline_sentiment','text'])
    airline_text = np.array(airline_data.text)
    airline_sentiment = np.array(airline_data.airline_sentiment)
    count = 0
    for i in range(len(airline_text)):
        if(count > 1000):
            break
        if(airline_sentiment[i] == "neutral"):
             neutral = np.append(neutral,airline_text[i])
             count+=1
    models = []
    models = np.append(models,load_model("ensemble_bgru.h5"))
    models = np.append(models,load_model("ensemble_gru.h5"))
    models = np.append(models,load_model("ensemble_gru.h5"))
    models = np.append(models,load_model("ensemble_lstm.h5"))
    models = np.append(models,load_model("ensemble_blstm.h5"))
    tokenizer = Tokenizer(num_words=300000)
    tokenizer.fit_on_texts(all_texts)

    yhats = [model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences(positive[:1000]),maxlen=75)) for model in models]
    yhats = np.array(yhats)
    summed = np.sum(yhats, axis=0)
    result = np.argmax(summed, axis=1)
    true_positive = np.count_nonzero(result == 1)
    yhats = [model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences(motivated[:1000]),maxlen=75)) for model in models]
    yhats = np.array(yhats)
    summed = np.sum(yhats, axis=0)
    result = np.argmax(summed, axis=1)
    true_motivated = np.count_nonzero(result == 2)
    yhats = [model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences(neutral[:1000]),maxlen=75)) for model in models]
    yhats = np.array(yhats)
    summed = np.sum(yhats, axis=0)
    result = np.argmax(summed, axis=1)
    true_neutral = np.count_nonzero(result == 0)
    yhats = [model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences(negative[:1000]),maxlen=75)) for model in models]
    yhats = np.array(yhats)
    summed = np.sum(yhats, axis=0)
    result = np.argmax(summed, axis=1)
    true_negative = np.count_nonzero(result == 4)
    yhats = [model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences(struggling[:1000]),maxlen=75)) for model in models]
    yhats = np.array(yhats)
    summed = np.sum(yhats, axis=0)
    result = np.argmax(summed, axis=1)
    true_strug = np.count_nonzero(result == 3)
    print("pos ",(true_positive/1000), "mot ",(true_motivated/1000), "neutral ",(true_neutral/1000)
       , "neg ",(true_negative/1000), "strug ",(true_strug/1000))