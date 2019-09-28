from keras import Input,Model
from keras.layers import LSTM,Bidirectional,Embedding,Dropout,Dense
# from CRFlayer import CRF
from CRFlayer_2 import CRF
from keras.optimizers import Adam
from Data import Dataloader
import os,numpy as np
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint,EarlyStopping
class Mymodel(object):
    def __init__(self,vocab_size,hidden,dropout_rate,use_word2vec=True,embedding_matrxi=None,depth=1,embedding_dim=100):
        self.vocab_size=vocab_size
        self.hidden=hidden
        self.dropout_rate=dropout_rate
        self.depth=depth
        self.use_wordvec=use_word2vec
        self.embedding_matrix=embedding_matrxi
        self.embedding_dim=embedding_dim

    def build_network(self,use_crf=False):
        input_layer=Input(shape=(None,),name='inputs')
        # x=Embedding(self.vocab_size,64)(input_layer) # wing.nus 64 | next hidden
        if self.use_wordvec==False:
            x = Embedding(self.vocab_size, self.hidden)(input_layer)
        else:
            x = Embedding(self.vocab_size+1, self.embedding_dim, weights=[self.embedding_matrix], trainable=False)(
                input_layer)
        x=Dropout(self.dropout_rate)(x)
        for i in range(self.depth):
            x = Bidirectional(LSTM(self.hidden, return_sequences=True),merge_mode='concat')(x)
        crf = CRF()
        if use_crf:
            x=Dense(self.hidden,activation='relu')(x)
            x=Dense(1,activation='sigmoid')(x)
            x =crf(x)
        else:
            x=Dense(self.hidden,activation='relu')(x)
            x=Dense(1,activation='sigmoid')(x)
        model=Model(input_layer,x)
        if use_crf == False:
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
        else:
            model.compile(optimizer=Adam(0.001),loss=crf.loss,metrics=[crf.accuracy])
        model.summary()
        return model

def train(use_crf=False,use_word2vec=True):
    model_name='use_word2vec_bilstmcrf-depth_1-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5' if use_word2vec else 'bilstmcrf-depth_1-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5'
    data=Dataloader()
    model=Mymodel(vocab_size=data.char_num,hidden=256,dropout_rate=0.3,depth=1,embedding_dim=data.embedding_dim,
                  use_word2vec=use_word2vec,
                  embedding_matrxi=None if use_word2vec==False else data.embedding_matrix
                  ).build_network(use_crf=use_crf)
    model.load_weights('models/use_word2vec_bilstmcrf-depth_1-001--0.34647--0.88836.hdf5')
    model.fit_generator(
        generator=data.generator(is_train=True),
        steps_per_epoch=data.steps_per_epoch,
        validation_data=data.generator(is_train=False),
        validation_steps=data.val_steps_per_epoch,
        verbose=1,
        initial_epoch=1,
        epochs=100,
        class_weight=[1,6.5] if use_crf==False else None,
        callbacks=[
            TensorBoard('logs'),
            ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1),
            EarlyStopping(monitor='val_loss',patience=28,verbose=1,restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join('models',model_name),verbose=1,save_weights_only=False,
                            save_best_only=True)
        ]
    )
def predict(use_crf=False,model_path='models/bilstmcrf-depth_1-006--0.36095--0.88239.hdf5'):
    data=Dataloader()
    model = Mymodel(vocab_size=data.char_num, hidden=256, dropout_rate=0.3, depth=1).build_network(use_crf=use_crf)
    model.load_weights(model_path)
    result=open('data/result_SI_2.txt','w')
    #articleid\tindex\tindex\n
    for article in os.listdir(data.dev_articles):
        path=os.path.join(data.dev_articles,article)
        f=open(path,'r',encoding='utf-8').read()
        inputs=[]
        for i in f:
            inputs.append(data.SI_token.word_index.get(i,0))
        inputs=np.array(inputs)
        inputs=np.expand_dims(inputs,axis=0)
        try:
            pred = model.predict(inputs)
            prob=pred[0,:,0]
            dicts={i:prob[i] for i in range(len(prob))}
            dicts=sorted(dicts.items(),key=lambda item:item[1])
            #0.866
            zero_num=int(0.866*len(prob))+1
            mask=[0]*len(prob)
            for i in range(zero_num,len(prob)):
                mask[dicts[i][0]]=1
            list=[]
            temp=[]
            for i in range(len(mask)):
                if mask[i]==1:
                    temp.append(i)
                else:
                    if temp!=[] and len(temp)>=2:
                        list.append([temp[0],temp[-1]])
                        result.write(article.strip('article').strip('.txt')+'\t'+str(temp[0])+'\t'+str(temp[-1])+'\n')
                    temp=[]
            print(article+' Done!!')

        except:
            print('error '+article)
    result.close()
# train(use_crf=False)
predict(use_crf=False,model_path='models/use_word2vec_bilstmcrf-depth_1-010--0.36705--0.86884.hdf5')
# train(use_crf=False,use_word2vec=True)

