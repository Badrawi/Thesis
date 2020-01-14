import string
import spacy
import tensorflow as tf
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from csv_reader import CSVReader
from text_preprocessing import TextPreprocessing
import re
import math
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from vent_api import VentApi
from models import Models
import traceback
from typing import Optional
import os
from logger_methods import setup_logger
import json
from tensorflow.keras import backend as K
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.python.keras.callbacks import TensorBoard
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

def my_model(sess):
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
        texts = []
        sentiments = []
       
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


        np.save(text_cache, texts)
        np.save(sentiment_cahce, sentiments)
    text_embedding = []
    sequences, word_index = get_word_index(texts)
    categorical_sentiments = to_categorical(sentiments, num_classes=5)
    X_train, X_test, Y_train, Y_test = train_test_split(texts, categorical_sentiments, test_size=0.2)
    train_input_ids, train_input_masks, train_segment_ids, train_labels = [],[],[],[]
    test_input_ids, test_input_masks, test_segment_ids, test_labels = [],[],[],[]
    if os.path.isfile('train.npz') and os.path.isfile('test.npz'):
        train = np.load('train.npz', allow_pickle=True)
        test = np.load('test.npz', allow_pickle=True)
        test_input_ids = test["test_input_ids"]
        test_input_masks = test["test_input_masks"]
        test_segment_ids = test["test_segment_ids"]
        test_labels = test["test_labels"]
        train_input_ids = train["train_input_ids"]
        train_input_masks = train["train_input_masks"]
        train_segment_ids = train["train_segment_ids"]
        train_labels = train["train_labels"]
    else:
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
        tokenizer = create_tokenizer_from_hub_module(bert_path,sess)
        train_examples = convert_text_to_examples(X_train, Y_train)
        test_examples = convert_text_to_examples(X_test, Y_test)
        (train_input_ids, train_input_masks, train_segment_ids, train_labels
        ) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=150)
        (test_input_ids, test_input_masks, test_segment_ids, test_labels
        ) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=150)
        np.savez('train.npz', train_input_ids=train_input_ids, train_input_masks=train_input_masks, 
        train_segment_ids=train_segment_ids, train_labels=train_labels)
        np.savez('test.npz',test_input_ids=test_input_ids, test_input_masks=test_input_masks, 
        test_segment_ids=test_segment_ids, test_labels=test_labels)
    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    models.build_myModel()
    
    fit_history = models.model.fit( X_train,
        Y_train,
        validation_data=(
            X_test,
            Y_test,
        ),
            batch_size=32, epochs=1, shuffle=True,callbacks=[tensorboard_callback])
    loss_history = fit_history.history["loss"]
    numpy_loss_history = np.array(loss_history)
    np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")
    result = models.model.predict_on_batch(pad_sequences(tokenizer.texts_to_sequences([
        " What happened 2 ur vegan food options?! At least say on ur site so i know I won't be able 2 eat anything for next 6 hrs #fail",
        " I sleep hungry and It gets harder everyday",
        "everything is great, i have lost some weight",
        "awesome, really cool",
        "should I play cards",
        "I am full and inshape",
        "is it okay to be that hungry at night?"])
        , maxlen=MAX_SEQUENCE_LENGTH))
    get_vents_logger.info("result: ", np.argmax(result, axis=-1), "\n")
    print("result: ", np.argmax(result, axis=-1), "\n")
def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    
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
    try:
       # tf.compat.v1.disable_eager_execution()
        sess = tf.Session()
        # initialize_vars(sess)  
        my_model(sess)
        # vader_model()
    except Exception as e:
        get_vents_logger.error(" Unexpected error: " + str(e) + traceback.format_exc())

