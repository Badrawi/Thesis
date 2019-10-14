from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D
from keras.layers import Input,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from keras import backend as K
from tensorflow.nn import space_to_depth
class RNN:

    embedding_dim = 500
    input_length = 100
    lstm_units = 128
    lstm_dropout = 0.1
    recurrent_dropout = 0.1
    spatial_dropout = 0.2
    filters = 64
    kernel_size = 3
    max_sequence_length = 150
    def myAct(self,out):
        return K.softmax(K.tanh(out))

    def build_lstm_model(self,embedding_matrix):
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
        self.model.add(Dense(5,activation='softmax'))
        self.model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

    def build_compound_model(self,embedding_matrix):
        sequence_input = Input(shape=(self.max_sequence_length,))
        embedding =  Embedding(
                input_dim=embedding_matrix.shape[0],
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=self.max_sequence_length,
                trainable=False,
            )(sequence_input)
        base = SpatialDropout1D(self.spatial_dropout)(embedding)
        base = Bidirectional(
                LSTM(self.lstm_units, return_sequences=True,
                     dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout)
            )(base)
        base = Conv1D(self.filters, kernel_size=self.kernel_size, padding='valid',
                   kernel_initializer='glorot_uniform')(base)
        avg_pool = GlobalAveragePooling1D()(base)
        max_pool = GlobalMaxPooling1D()(base)
        concat_out = concatenate([avg_pool,max_pool])
        # self.model = Sequential()
        # self.model.add(concatenate([avg_pool,max_pool]))
        pred = Dense(5, activation='softmax')(concat_out)
        self.model = Model(sequence_input,pred)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])