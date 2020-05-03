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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from models import Models
from tensorflow.keras.models import load_model
import os
import json
# Create our list of punctuation marks
punctuations = string.punctuation

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
models = Models()
tokenizer = Tokenizer(num_words=100000)
# Creating our tokenizer function
def get_word_index(texts):
    # Creating our token object, which is used to create documents with linguistic annotations.
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    sequences = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
    return sequences,word_index

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
    sentiment_cache = "sentiment.npy"
    text_cache = "text.npy"
    text_embedding_cache= "text_embedding2.npy"
    airline_data = CSVReader.dataframe_from_file("Tweets.csv",['airline_sentiment','text'])
    stress_data = CSVReader.dataframe_from_file("twitter.csv", ['original_text'])
    progress_data = CSVReader.dataframe_from_file("fit.csv", ['original_text'])
    texts = np.array(airline_data.text)
    sentiments = np.array(airline_data.airline_sentiment)
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
    for text in stress_data:
        texts = np.append(texts,text)
        sentiments = np.append(sentiments,-2)
    for text in progress_data:
        texts = np.append(texts,text)
        sentiments = np.append(sentiments,1)
    np.save(sentiment_cache,sentiments)
    np.save(text_cache,texts)
    sequences , word_index = get_word_index(texts)
    categorical_sentiments = to_categorical(sentiments,num_classes=4)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)
    text_embedding = np.zeros((len(word_index)+1,300))
    for word,i in word_index.items():
        text_embedding[i] = nlp(word).vector
        
    np.save(text_embedding_cache,text_embedding)
    # models.build_myModel(text_embedding,False)
    models.build_LSTM_model(text_embedding)
    models.model.fit(pad_sequences(tokenizer.texts_to_sequences(X_train),maxlen=MAX_SEQUENCE_LENGTH),
                       Y_train,
                       batch_size=512,epochs=10,validation_data=(pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=MAX_SEQUENCE_LENGTH)
                                                                 ,Y_test),shuffle=True)

    result = models.model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences([
                                                                               " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
                                                                                     " I sleep hungry and It gets harder everyday",
                                                                                     "everything is great, i have lost some weight",
                                                                                     "awesome, really cool",
                                                                                     "should I play cards",
                                                                                     "I am full and inshape",
                                                                                     "is it okay to be that hungry at night?"])
                                                 ,maxlen=MAX_SEQUENCE_LENGTH))
    print("result: ", np.argmax(result,axis=-1),"\n")