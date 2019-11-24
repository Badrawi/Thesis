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
from models import Models
import traceback
from typing import Optional
import multiprocessing
from keras.models import load_model
import os
from logger_methods import setup_logger
import json
from keras import backend as K
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bert_keras import create_tokenizer_from_hub_module, convert_text_to_examples, convert_examples_to_features
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
models = Models()
tokenizer = Tokenizer(num_words=100000)
analyser = SentimentIntensityAnalyzer()
# Creating our tokenizer function
def get_word_index(texts):
    # Creating our token object, which is used to create documents with linguistic annotations.
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    sequences = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
    return sequences,word_index
def print_sentiment_scores(sentences):
    for sentence in sentences:
        snt = analyser.polarity_scores(sentence)
        print("{:-<40} {}".format(sentence, str(snt)))
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

def my_model():
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # print("get gpus ",K.tensorflow_backend._get_available_gpus())
    sentiment_cahce = "sentiment_cache.npy"
    text_cache = "text_cache.npy"
    text_embedding_cache = "text_embedding.npy"
    sentiments = []
    texts = []
    if os.path.isfile(sentiment_cahce) and os.path.isfile(text_cache):
        sentiments = np.load(sentiment_cahce, allow_pickle=True)
        texts = np.load(text_cache, allow_pickle=True)
    else:
        # airline_data = CSVReader.dataframe_from_file("Tweets.csv", ['airline_sentiment', 'text'])
        # stress_data = CSVReader.dataframe_from_file("twitter.csv", ['original_text']).original_text
        # progress_data = CSVReader.dataframe_from_file("fit.csv", ['original_text']).original_text
        # texts = np.array(airline_data.text)
        # sentiments = np.array(airline_data.airline_sentiment)
        texts = []
        sentiments = []
        # for i in range(len(sentiments)):
        #     texts[i] = textPreProcessing.remove_special_characters(texts[i], True)
        #     texts[i] = textPreProcessing.remove_accented_chars(texts[i])
        #     texts[i] = textPreProcessing.remove_whiteList(texts[i])
        #     if (sentiments[i] == "neutral"):
        #         sentiments[i] = 0
        #     elif(sentiments[i] == "positive"):
        #         sentiments[i] = 1
        #     else:
        #         sentiments[i] = -1
        # with multiprocessing.Pool() as pool:
        #     positive = pool.starmap(getVentsSentiment, zip(vent_positive))
        #     negative = pool.starmap(getVentsSentiment, zip(vent_negative))
        good = ventApi.getVents(ventApi.EMOTION_GOOD_ID)
        energized = ventApi.getVents(ventApi.EMOTION_ENERGIZED_ID)
        bad = ventApi.getVents(ventApi.EMOTION_BAD_ID)
        struggle = ventApi.getVents(ventApi.EMOTION_STRUGGLE_ID)
        neutral = ventApi.getVents(ventApi.EMOTION_NEUTRAL_ID)
        texts = np.append(texts, good)
        good_sentiments = [1] * len(good)
        sentiments = np.append(sentiments, good_sentiments)
        energized_sentiments = [2] * len(energized)
        texts = np.append(texts, energized)
        sentiments = np.append(sentiments, energized_sentiments)
        texts = np.append(texts, bad)
        bad_sentiments = [-1] * len(bad)
        sentiments = np.append(sentiments, bad_sentiments)
        texts = np.append(texts, struggle)
        struggle_sentiments = [-2] * len(struggle)
        sentiments = np.append(sentiments, struggle_sentiments)
        texts = np.append(texts, neutral)
        neutral_sentiments = [0] * len(neutral)
        sentiments = np.append(sentiments, neutral_sentiments)
        # for text in stress_data:
        #     text = textPreProcessing.remove_special_characters(text, True)
        #     text = textPreProcessing.remove_accented_chars(text)
        #     text = textPreProcessing.remove_whiteList(text)
        #     texts = np.append(texts,[text])
        #     sentiments = np.append(sentiments,[-2])
        # for text in progress_data:
        #     text = textPreProcessing.remove_special_characters(text, True)
        #     text = textPreProcessing.remove_accented_chars(text)
        #     text = textPreProcessing.remove_whiteList(text)
        #     texts = np.append(texts,[text])
        #     sentiments = np.append(sentiments,[1])

        np.save(text_cache, texts)
        np.save(sentiment_cahce, sentiments)
    text_embedding = []
    sequences, word_index = get_word_index(texts)
    categorical_sentiments = to_categorical(sentiments, num_classes=5)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    tokenizer = create_tokenizer_from_hub_module(bert_path)
    train_examples = convert_text_to_examples(X_train, Y_train)
    test_examples = convert_text_to_examples(X_test, Y_test)
    (train_input_ids, train_input_masks, train_segment_ids, train_labels
     ) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=150)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels
     ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=150)
    # if os.path.isfile(text_embedding_cache):
    #     text_embedding = np.load(text_embedding_cache, allow_pickle=True)
    # else:
    #     text_embedding = np.zeros((len(word_index) + 1, 300))
    #     for word, i in word_index.items():
    #         text_embedding[i] = nlp(word).vector
    #     np.save(text_embedding_cache, text_embedding)
    models.build_myModel(text_embedding)
    models.model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    models.model.fit(pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_SEQUENCE_LENGTH),
              Y_train,
              batch_size=512, epochs=5,
              validation_data=(pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_SEQUENCE_LENGTH)
                               , Y_test), shuffle=True)

    result = models.model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences([
        " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
        " I sleep hungry and It gets harder everyday",
        "everything is great, i have lost some weight",
        "awesome, really cool",
        "should I play cards",
        "I am full and inshape",
        "is it okay to be that hungry at night?"])
        , maxlen=MAX_SEQUENCE_LENGTH))

    print("result: ", np.argmax(result, axis=-1), "\n")
def vader_model():
    print_sentiment_scores([
        " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
        " I sleep hungry and It gets harder everyday",
        "everything is great, i have lost some weight",
        "awesome, really cool",
        "should I play cards",
        "I am full and inshape",
        "is it okay to be that hungry at night?"])
if __name__ == "__main__":
    my_model()
    # vader_model()

