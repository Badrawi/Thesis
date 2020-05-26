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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def test_all_models():
    # sentiments = np.load('sentiment_cache.npy', allow_pickle=True)
    texts = np.load('text_cache.npy', allow_pickle=True)
    sentiments = np.load('sentiment_cache.npy', allow_pickle=True)
    categorical_sentiments = to_categorical(sentiments,num_classes=5)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    model = load_model("savedModel/saved-model3-15.h5")
    # result = model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences([
    #                                                                            " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
    #                                                                                  " I sleep hungry and It gets harder everyday",
    #                                                                                  "everything is great, i have lost some weight",
    #                                                                                  "awesome, really cool",
    #                                                                                  "should I play cards",
    #                                                                                  "I am full and inshape",
    #                                                                                  "is it okay to be that hungry at night?"])
    #                                              ,maxlen=75))
    # print("result: ", np.argmax(result,axis=-1),"\n")
    

def train_BLSTM():
    embeddings = np.load('text_embedding.npy', allow_pickle=True)
    sentiments = np.load('sentiments.npy', allow_pickle=True)
    texts = np.load('texts.npy', allow_pickle=True)
    all_texts = np.load('text_cache.npy', allow_pickle=True)
    categorical_sentiments = to_categorical(sentiments,num_classes=5)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_texts)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)
    models = Models()
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    filepath =  "blstm.h5"
    models.build_BLSTM_model(embeddings)
    model = models.model
    if os.path.isfile(filepath):
        model = load_model(filepath)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
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

def train_LSTM():
    embeddings = np.load('text_embedding.npy', allow_pickle=True)
    sentiments = np.load('sentiments.npy', allow_pickle=True)
    texts = np.load('texts.npy', allow_pickle=True)
    all_texts = np.load('text_cache.npy', allow_pickle=True)
    categorical_sentiments = to_categorical(sentiments,num_classes=5)
    tokenizer = Tokenizer(num_words=300000)
    tokenizer.fit_on_texts(all_texts)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)
    models = Models()
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    filepath =  "lstm.h5"
    models.build_LSTM_model(embeddings)
    model = models.model
    if os.path.isfile(filepath):
        model = load_model(filepath)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
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


def test_vader():
    print("vader")
    sentiments = np.load('sentiments.npy', allow_pickle=True)
    texts = np.load('texts.npy', allow_pickle=True)
    count_positive = 0
    true_positive = 0
    count_motivated = 0
    true_motivated = 0
    count_neutral = 0
    true_neutral = 0
    count_negative = 0
    true_negative = 0
    count_strug = 0
    true_strug = 0
    analyzer = SentimentIntensityAnalyzer()
    for i in range(len(sentiments)):
        if(sentiments[i] == 1):
            count_positive += 1
            vs = analyzer.polarity_scores(texts[i])
            if(vs["compound"] > 0.05 and vs["compound"] < 0.55):
                true_positive += 1
        if(sentiments[i] == 2):
            count_motivated += 1
            vs = analyzer.polarity_scores(texts[i])
            if(vs["compound"] > 0.55):
                true_motivated += 1
        if(sentiments[i] == 0):
            count_neutral += 1
            vs = analyzer.polarity_scores(texts[i])
            if(vs["compound"] > -0.05 and vs["compound"] < 0.05):
                true_neutral += 1
        if(sentiments[i] == -1):
            count_negative += 1
            vs = analyzer.polarity_scores(texts[i])
            if(vs["compound"] > -0.55 and vs["compound"] < -0.05):
                true_negative += 1
        if(sentiments[i] == -2):
            count_strug += 1
            vs = analyzer.polarity_scores(texts[i])
            if(vs["compound"] < -0.55):
                true_strug += 1

    print("pos ",(true_positive/count_positive), "mot ",(true_motivated/count_motivated), "neutral ",(true_neutral/count_neutral)
    , "neg ",(true_negative/count_negative), "strug ",(true_strug/count_strug))

def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model_test():
    print("here")
    sentiments = np.load('sentiments.npy', allow_pickle=True)
    texts = np.load('texts.npy', allow_pickle=True)
    all_texts = np.load('text_cache.npy', allow_pickle=True)
    neutral = []

    _, X_test, _, Y_test = train_test_split(texts, sentiments, test_size=0.01)
    print("here ",len(Y_test))
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
    X_test = np.append(X_test,neutral)
    Y_test = np.append(Y_test,[0]*len(neutral))
    categ_test = to_categorical(Y_test,num_classes=5)
    tokenizer = Tokenizer(num_words=300000)
    tokenizer.fit_on_texts(all_texts)
    model = load_model("savedModel2/saved-model3-60.h5")
    result = model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=75))
    result = np.argmax(result,axis=-1)
    cat_result = to_categorical(result,num_classes=5)
    print("f1 ",f1(categ_test,cat_result))

if __name__ == "__main__":
    # train_BLSTM()
    # train_LSTM()
    # test_all_models()
    # test_vader()
    model_test()
