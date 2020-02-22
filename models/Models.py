from tensorflow.keras.layers import Input,Dense,LSTM,GRU,Bidirectional,Embedding,Dropout,TimeDistributed,Activation
from tensorflow.keras import Model
from models.CRF import CRF
class CustomModels(object):
    def __init__(self,model_name,vocab_size,embedding_name=None):
        self.model_name=model_name
        self.embedding_name=embedding_name
        self.vocab_size=vocab_size
    def build_model(self):
        if self.embedding_name is None:
            Input_layer=Input(shape=(None,),name='Input_layer')
            x=Embedding(self.vocab_size,100)(Input_layer)
            if self.model_name.lower()=='lstm':
                x=Bidirectional(LSTM(128,return_sequences=True),merge_mode='concat',name='bilstm')(x)
                x=Dropout(0.4,name='lstm_dropout')(x)
                x=TimeDistributed(Dense(1),name='Time_Dense')(x)
                x=Activation('sigmoid')(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,'binary_crossentropy',['acc']
            elif self.model_name.lower()=='gru':
                x=Bidirectional(GRU(128,return_sequences=True),merge_mode='concat',name='bigru')(x)
                x=Dropout(0.4,name='lstm_dropout')(x)
                x=TimeDistributed(Dense(1),name='Time_Dense')(x)
                x=Activation('sigmoid')(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,'binary_crossentropy',['acc']
            elif self.model_name.lower()=='lstm-crf':
                x=Bidirectional(LSTM(128,return_sequences=True),merge_mode='concat',name='bilstm')(x)
                x=Dense(64,activation='tanh',name='dense_layer')(x)
                x=Dense(1,name='dense_for_crf',activation='sigmoid')(x)
                crf_layer=CRF(1,name='crf')
                x=crf_layer(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,crf_layer.loss,[crf_layer.viterbi_accuracy]
            else:
                x=Bidirectional(GRU(128,return_sequences=True),merge_mode='concat',name='bigru')(x)
                x=Dense(64,activation='tanh',name='dense_layer')(x)
                x=Dense(1,name='dense_for_crf',activation='sigmoid')(x)
                crf_layer=CRF(1,name='crf')
                x=crf_layer(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,crf_layer.loss,[crf_layer.viterbi_accuracy]

if __name__=='__main__':
    m=CustomModels(model_name='lstm-crf',vocab_size=147,embedding_name=None)
    m.build_model()
