import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from csv_reader import CSVReader
from text_preprocessing import TextPreprocessing
import re
import math
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from vent_api import VentApi
from rnn import RNN
import traceback
from typing import Optional
import multiprocessing
from keras.models import load_model
import os
from logger_methods import setup_logger
import json

get_vents_logger = setup_logger('get_vents', 'extract_progress.log')
# Create our list of punctuation marks
punctuations = string.punctuation
ventApi = VentApi()
# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS
sentiment_labels = []
texts = []
text_embedding = []
# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()
nlp = spacy.load('en_core_web_lg')
textPreProcessing  = TextPreprocessing()
MAX_SEQUENCE_LENGTH = 150
rnnModel = RNN()
tokenizer = Tokenizer(num_words=100000)
# Creating our tokenizer function
def get_word_index(texts):
    # Creating our token object, which is used to create documents with linguistic annotations.
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    sequences = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
    return sequences,word_index
def getVentsSentiment(vents):
    count = 0
    texts =[]
    for text in vents:
        try:
            if count % 1000 == 0:
                get_vents_logger.info("[COUNT: " + str(count) + "]")
            text = textPreProcessing.remove_special_characters(text, True)
            text = textPreProcessing.remove_accented_chars(text)
            text = textPreProcessing.remove_whiteList(text)
            texts = np.append(texts, [text])
        except Exception as e:
            get_vents_logger.error("for text: ",text," Unexpected error: " + str(e) + traceback.format_exc())
        count += 1
    return texts
def get_embeddings(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in STOP_WORDS and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

if __name__ == "__main__":

    airline_data = CSVReader.dataframe_from_file("Tweets.csv",['airline_sentiment','text'])
    stress_data = CSVReader.dataframe_from_file("twitter.csv", ['original_text']).original_text
    progress_data = CSVReader.dataframe_from_file("fit.csv", ['original_text']).original_text
    texts = np.array(airline_data.text)
    sentiments = np.array(airline_data.airline_sentiment)
    sentiment_cahce = "sentiment_cache.npy"
    text_cache = "text_cache.npy"
    vent_positive = np.array(ventApi.getPositiveVents())
    vent_negative = np.array(ventApi.getNegativeVents())
    if os.path.isfile(sentiment_cahce) and os.path.isfile(text_cache):
        sentiments = np.load(sentiment_cahce)
        texts = np.load(text_cache)
    else:
        for i in range(len(sentiments)):
            texts[i] = textPreProcessing.remove_special_characters(texts[i], True)
            texts[i] = textPreProcessing.remove_accented_chars(texts[i])
            texts[i] = textPreProcessing.remove_whiteList(texts[i])
            if (sentiments[i] == "neutral"):
                sentiments[i] = 0
            elif(sentiments[i] == "positive"):
                sentiments[i] = 1
            else:
                sentiments[i] = -1
        # with multiprocessing.Pool() as pool:
        #     positive = pool.starmap(getVentsSentiment, zip(vent_positive))
        #     negative = pool.starmap(getVentsSentiment, zip(vent_negative))
        positive = ventApi.getPositiveVents()
        negative = ventApi.getNegativeVents()
        np.append(texts,positive)
        pos_sentiments = [1]*len(positive)
        np.append(sentiments,pos_sentiments)
        neg_sentiments = [-1] * len(negative)
        np.append(texts, negative)
        np.append(sentiments, neg_sentiments)

        for text in stress_data:
            text = textPreProcessing.remove_special_characters(text, True)
            text = textPreProcessing.remove_accented_chars(text)
            text = textPreProcessing.remove_whiteList(text)
            texts = np.append(texts,[text])
            sentiments = np.append(sentiments,[-2])
        for text in progress_data:
            text = textPreProcessing.remove_special_characters(text, True)
            text = textPreProcessing.remove_accented_chars(text)
            text = textPreProcessing.remove_whiteList(text)
            texts = np.append(texts,[text])
            sentiments = np.append(sentiments,[1])

        np.save(text_cache,texts)
        np.save(sentiment_cahce, sentiments)

    sequences , word_index = get_word_index(texts)
    categorical_sentiments = to_categorical(sentiments,num_classes=4)
    print("category: ",categorical_sentiments)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)
    text_embedding = np.zeros((len(word_index)+1,300))
    for word,i in word_index.items():
        text_embedding[i] = nlp(word).vector
    rnnModel.build_model(1, text_embedding)
    rnnModel.model.fit(pad_sequences(tokenizer.texts_to_sequences(X_train),maxlen=MAX_SEQUENCE_LENGTH),
                       Y_train,
                       batch_size=512,epochs=10,validation_data=(pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=MAX_SEQUENCE_LENGTH)
                                                                 ,Y_test),shuffle=True)

    result = rnnModel.model.predict_classes(pad_sequences(tokenizer.texts_to_sequences([
                                                                               " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
                                                                                     " I sleep hungry and It gets harder everyday",
                                                                                     "everything is great, i have lost some weight",
                                                                                     "awesome, really cool",
                                                                                     "should I play cards",
                                                                                     "I am full and inshape",
                                                                                     "is it okay to be that hungry at night?"])
                                                 ,maxlen=MAX_SEQUENCE_LENGTH))
    print("result: ",result,"\n")
