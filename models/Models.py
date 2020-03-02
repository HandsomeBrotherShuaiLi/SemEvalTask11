from models.CRF import CRF
from keras_bert import load_trained_model_from_checkpoint
class CustomModels(object):
    def __init__(self,model_name,vocab_size,embedding_name=None,layer_number=2,paths=None):
        self.model_name=model_name
        self.embedding_name=embedding_name
        self.vocab_size=vocab_size
        self.layer_number=layer_number
        self.paths=paths
    def build_model(self,trainable=True):
        if self.embedding_name is None:
            from tensorflow.keras.layers import Input,Dense,LSTM,GRU,Bidirectional,Embedding,Dropout,TimeDistributed,Activation
            from tensorflow.keras import Model
            Input_layer=Input(shape=(None,),name='Input_layer')
            embedd_layer=Embedding(self.vocab_size,100)
            x=embedd_layer(Input_layer)
            if self.model_name.lower()=='lstm':
                x=Dropout(0.4,name='lstm_dropout')(x)
                for i in range(self.layer_number):
                    x=Bidirectional(LSTM(128,return_sequences=True),merge_mode='concat',name='bilstm_{}'.format(i))(x)
                x=TimeDistributed(Dense(1),name='Time_Dense')(x)
                x=Activation('sigmoid')(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,'binary_crossentropy',['acc'],embedd_layer
            elif self.model_name.lower()=='gru':
                x=Dropout(0.4,name='gru_dropout')(x)
                for i in range(self.layer_number):
                    x=Bidirectional(GRU(128,return_sequences=True),merge_mode='concat',name='bigru_{}'.format(i))(x)
                x=TimeDistributed(Dense(1),name='Time_Dense')(x)
                x=Activation('sigmoid')(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,'binary_crossentropy',['acc'],embedd_layer
            elif self.model_name.lower()=='lstm-crf':
                x=Dropout(0.4,name='lstm_dropout')(x)
                for i in range(self.layer_number):
                    x=Bidirectional(LSTM(128,return_sequences=True),merge_mode='concat',name='bilstm_{}'.format(i))(x)
                x=Dense(64,activation='tanh',name='dense_layer')(x)
                x=Dense(1,name='dense_for_crf',activation='sigmoid')(x)
                crf_layer=CRF(1,name='crf')
                x=crf_layer(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,crf_layer.loss,[crf_layer.viterbi_accuracy],embedd_layer
            else:
                x=Dropout(0.4,name='gru_dropout')(x)
                for i in range(self.layer_number):
                    x=Bidirectional(GRU(128,return_sequences=True),merge_mode='concat',name='bigru_{}'.format(i))(x)
                x=Dense(64,activation='tanh',name='dense_layer')(x)
                x=Dense(1,name='dense_for_crf',activation='sigmoid')(x)
                crf_layer=CRF(1,name='crf')
                x=crf_layer(x)
                model=Model(Input_layer,x)
                model.summary()
                return model,crf_layer.loss,[crf_layer.viterbi_accuracy],embedd_layer
        else:
            from keras.layers import Input,Dense,LSTM,GRU,Bidirectional,Embedding,Dropout,TimeDistributed,Activation
            from keras import Model
            assert self.paths is not None
            config_path,checkpoints_path,vocab_path=self.paths
            bert_model=load_trained_model_from_checkpoint(
                config_file=config_path,
                checkpoint_file=checkpoints_path,
                training=trainable
            )
            inputs=bert_model.inputs[:2]
            x = bert_model.layers[-1].output if trainable==False else bert_model.get_layer(name='Encoder-24-FeedForward-Norm').output
            x=Dropout(0.4)(x)
            if self.model_name.lower()=='lstm':
                for i in range(self.layer_number):
                    x=Bidirectional(LSTM(128,return_sequences=True),merge_mode='concat',name='bilstm_{}'.format(i))(x)
                x=TimeDistributed(Dense(1),name='Time_Dense')(x)
                x=Activation('sigmoid')(x)
                model=Model(inputs,x)
                model.summary()
                return model,'binary_crossentropy',['acc'],bert_model
            elif self.model_name.lower()=='gru':
                for i in range(self.layer_number):
                    x=Bidirectional(GRU(128,return_sequences=True),merge_mode='concat',name='bigru_{}'.format(i))(x)
                x=TimeDistributed(Dense(1),name='Time_Dense')(x)
                x=Activation('sigmoid')(x)
                model=Model(inputs,x)
                model.summary()
                return model,'binary_crossentropy',['acc'],bert_model
            elif self.model_name.lower()=='lstm-crf':
                for i in range(self.layer_number):
                    x=Bidirectional(LSTM(128,return_sequences=True),merge_mode='concat',name='bilstm_{}'.format(i))(x)
                x=Dense(64,activation='tanh',name='dense_layer')(x)
                x=Dense(1,name='dense_for_crf',activation='sigmoid')(x)
                crf_layer=CRF(1,name='crf')
                x=crf_layer(x)
                model=Model(inputs,x)
                model.summary()
                return model,crf_layer.loss,[crf_layer.viterbi_accuracy],bert_model
            else:
                for i in range(self.layer_number):
                    x=Bidirectional(GRU(128,return_sequences=True),merge_mode='concat',name='bigru_{}'.format(i))(x)
                x=Dense(64,activation='tanh',name='dense_layer')(x)
                x=Dense(1,name='dense_for_crf',activation='sigmoid')(x)
                crf_layer=CRF(1,name='crf')
                x=crf_layer(x)
                model=Model(inputs,x)
                model.summary()
                return model,crf_layer.loss,[crf_layer.viterbi_accuracy],bert_model

if __name__=='__main__':
    m=CustomModels(model_name='lstm',vocab_size=147,embedding_name='bert',
                   paths=['/Users/shuai.li/PycharmProjects/SemEvalTask11/pretrained_embedding/wwm_uncased_L-24_H-1024_A-16/bert_config.json',
                          '/Users/shuai.li/PycharmProjects/SemEvalTask11/pretrained_embedding/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt',
                          '/Users/shuai.li/PycharmProjects/SemEvalTask11/pretrained_embedding/wwm_uncased_L-24_H-1024_A-16/vocab.txt'])
    m.build_model()
