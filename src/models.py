from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D,GRU
from keras.layers import Input,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from keras import backend as K
from attention_layer import AttentionDecoder
from bert_keras import BertLayer
import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow.nn import space_to_depth
class Models:

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


    def build_Base_model(self,embedding_matrix):
        self.sequence_input = Input(shape=(self.max_sequence_length,))
        embedding = Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=self.max_sequence_length,
            trainable=False,
        )(self.sequence_input)
        base = SpatialDropout1D(self.spatial_dropout)(embedding)
        return base
    def build_Base_Bert_model(self,input_id,input_mask,input_segment):
        tags = set()
        tags.add("train")
        in_id = Input(shape=input_id.shape, name="input_ids")
        in_mask = Input(shape=input_mask.shape, name="input_masks")
        in_segment = Input(shape=input_segment.shape, name="segment_ids")
        self.bert_inputs = [in_id, in_mask, in_segment]
    
        # Instantiate the custom Bert Layer defined above
        # bertlayer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
        #                     trainable=True)
        bert_output = BertLayer()(self.bert_inputs)
        print("********print bert******")
        print(bert_output.shape)
        base = SpatialDropout1D(self.spatial_dropout)(bert_output)
        return base
    def build_GRU_model(self,base):
        base = GRU(128, return_sequences=True)(base)
        base = GRU(128, return_sequences=True)(base)
        avg = GlobalAveragePooling1D()(base)
        max = GlobalMaxPooling1D()(base)
        return concatenate([avg, max])
    def build_BLTSM_model(self,base):
        base = Bidirectional(
                LSTM(self.lstm_units, return_sequences=True,
                     dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout),

            )(base)
        #base = AttentionDecoder(150, 50)(base)
        base = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True,
                 dropout=self.lstm_dropout, recurrent_dropout=self.recurrent_dropout)

        )(base)
        avg = GlobalAveragePooling1D()(base)
        max = GlobalMaxPooling1D()(base)
        return concatenate([avg, max])


    def build_CNN_model(self,base):
        base = Conv1D(self.filters, kernel_size=self.kernel_size, padding='valid',
                      kernel_initializer='glorot_uniform')(base)
      #  base = MaxPooling1D(pool_size=2)(base)
      #  base = Conv1D(self.filters, kernel_size=self.kernel_size, padding='valid',
        #              kernel_initializer='glorot_uniform')(base)
        avg = GlobalAveragePooling1D()(base)
        max = GlobalMaxPooling1D()(base)
        return concatenate([avg,max])

    def build_myModel(self,input_id,input_mask=None,input_segment=None,bert=True):
        base = None
        if(bert):
            base = self.build_Base_Bert_model(input_id,input_mask,input_segment)
        else:
            base = self.build_Base_model(input_id)
        print("base****** ",base)
        concat_cnn = self.build_CNN_model(base)
        concat_blstm = self.build_BLTSM_model(base)
        concat_gru = self.build_GRU_model(base)
        pred_cnn = Dense(128, activation='relu')(concat_cnn)
        pred_bltsm = Dense(128, activation='relu')(concat_blstm)
        pred_gru = Dense(128, activation='relu')(concat_gru)
        concat_out = concatenate([pred_cnn, pred_bltsm])
        concat_out = concatenate([concat_out, pred_gru])
        pred = Dense(5, activation='softmax')(concat_out)
        if(bert):
            self.model - Model(self.bert_inputs,pred)
        else:
            self.model = Model(self.sequence_input, pred)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
