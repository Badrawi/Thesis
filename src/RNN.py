from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D
from keras import backend as K
class RNN:

    def myAct(self,out):
        return K.softmax(K.tanh(out))

    def build_model(self,hidden_layers,embedding_matrix):
        self.model = Sequential()
        self.model.add(
            Embedding(input_dim=embedding_matrix.shape[0],
                      output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix],
                      input_length=150,
                      trainable=False))
        #self.model.add(SpatialDropout1D(0.2))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(LSTM(128,return_sequences=True))
        self.model.add(LSTM(128, return_sequences=True))
        # self.model.add(SpatialDropout1D(0.2))
        # self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        # self.model.add(LSTM(128, return_sequences=True))
        # self.model.add(LSTM(128, return_sequences=True))
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(4,activation='softmax'))
        self.model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

