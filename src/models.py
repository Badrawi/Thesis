from keras.models import Sequential,Model
from keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout2D,Conv2D,MaxPooling2D,GRU
from keras.layers import Input,Bidirectional,GlobalAveragePooling2D,GlobalMaxPooling2D,concatenate
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
        self.model.add(MaxPooling2D(pool_size=2))
        self.model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
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
        print("****shape*****")
        print(embedding.shape)
        base = SpatialDropout2D(self.spatial_dropout)(embedding)
        return base
    def build_Base_Bert_model(self):
        tags = set()
        tags.add("train")
        in_id = Input(shape=(self.max_sequence_length,), name="input_ids")
        in_mask = Input(shape=(self.max_sequence_length,), name="input_masks")
        in_segment = Input(shape=(self.max_sequence_length,), name="segment_ids")
        #self.bert_inputs = [in_id, in_mask, in_segment]
        bert_module = hub.Module("BERT_MODEL_HUB", tags=tags, trainable=True)
        self.bert_inputs = dict(
        input_ids=in_id,
        input_mask=in_mask,
        segment_ids=in_segment)
        bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True)
        output_layer = bert_outputs["sequence_output"]
        # Instantiate the custom Bert Layer defined above
        # bertlayer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
        #                     trainable=True)
       # bert_output = BertLayer(n_fine_tune_layers=10,)(self.bert_inputs)
        #print("********print bert******")
        #print(bert_output)
        base = SpatialDropout2D(self.spatial_dropout)(output_layer)
        return base
    def build_GRU_model(self,base):
        base = GRU(128, return_sequences=True)(base)
        base = GRU(128, return_sequences=True)(base)
        avg = GlobalAveragePooling2D()(base)
        max = GlobalMaxPooling2D()(base)
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
        avg = GlobalAveragePooling2D()(base)
        max = GlobalMaxPooling2D()(base)
        return concatenate([avg, max])


    def build_CNN_model(self,base):
        base = MaxPooling2D(pool_size=2)(base)
        base = Conv2D(self.filters, kernel_size=self.kernel_size, padding='valid',
                      kernel_initializer='glorot_uniform')(base)
        base = MaxPooling2D(pool_size=2)(base)
        base = Conv2D(self.filters, kernel_size=self.kernel_size, padding='valid',
                      kernel_initializer='glorot_uniform')(base)
        avg = GlobalAveragePooling2D()(base)
        max = GlobalMaxPooling2D()(base)
        return concatenate([avg,max])

    def build_myModel(self,embedding_matrix,bert=True):
        base = None
        if(bert):
            base = self.build_Base_Bert_model()
        else:
            base = self.build_Base_model(embedding_matrix)
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
