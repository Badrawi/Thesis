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


if __name__ == "__main__":
    embeddings = []
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
    filepath =  "savedModel2/saved-model3-{epoch:02d}.h5"
    filepath2 = "return.h5"
    model = load_model(filepath2)
    models.build_myModel(embeddings,model)
    model = models.model
    if os.path.isfile("savedModel/saved-model3-25.h5"):
        model = load_model("savedModel/saved-model3-25.h5")

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,tensorboard_callback]

    model.fit(pad_sequences(tokenizer.texts_to_sequences(X_train[:100000]),maxlen=150),
                      Y_train[:100000],
                       batch_size=512,epochs=10,validation_data=(pad_sequences(tokenizer.texts_to_sequences(X_test[:5000]),maxlen=150)
                                                                 ,Y_test[:5000]),callbacks=callbacks_list,shuffle=True)

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
