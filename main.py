import os
from keras.preprocessing.text import Tokenizer
import numpy as np
import keras.backend as K
import gensim
from keras import Input,Model
from keras.layers import LSTM,Bidirectional,Embedding,Dropout,Dense
# from CRFlayer import CRF
# from CRFlayer_2 import CRF
from keras.optimizers import Adam
import os
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint,EarlyStopping
class Dataloader(object):
    """
    字符级别的处理，单词级别的处理很难，暂时不做
    """
    def __init__(self,train_articles='data/train-articles',
                 dev_articles='data/dev-articles',
                 SI_dir='data/train-labels-task1-span-identification',
                 TC_dir='data/train-labels-task2-technique-classification',
                 split_ratio=0.2,
                 batch_size=3,
                 embedding_dim=100,
                 use_word2vec=True):
        self.train_articles = train_articles
        self.dev_articles=dev_articles
        self.SI_dir=SI_dir
        self.TC_dir=TC_dir
        self.embedding_dim=embedding_dim
        self.use_word2vec=use_word2vec
        self.batch_size=batch_size
        self.SI_token,self.embedding_matrix=self.preprocess_for_SI()
        all_index=np.array(range(len(os.listdir(self.train_articles))))
        val_num=int(all_index.shape[0]*split_ratio)
        self.val_index=np.random.choice(all_index,val_num,replace=False)
        self.train_index=np.array([i for i in all_index if i not in self.val_index])
        print('train num:{} val num:{} total num:{}'.format(len(self.train_index),
                                                            len(self.val_index),
                                                            len(all_index)))
        self.sort_length()
        self.steps_per_epoch=len(self.train_index)//self.batch_size
        self.val_steps_per_epoch=len(self.val_index)//self.batch_size
        self.char_num=len(self.SI_token.word_index)
        # word_index -> word:index

    def preprocess_for_SI(self,wordembedding=False):
        token=Tokenizer(num_words=40000,char_level=True,lower=False)
        sentence=[]
        zero_num=0
        one_num=0
        label_lens=[]
        s=[]
        for article in os.listdir(self.train_articles):
            path=os.path.join(self.train_articles,article)
            f=open(path,'r',encoding='utf-8').read()
            sentence+=f
            zero_num+=len(f)
            s.append(list(f))
            labels=open(os.path.join(self.SI_dir,article.replace('.txt','.task1-SI.labels')),'r',encoding='utf-8').readlines()
            for i in labels:
                temp=i.split('\t')
                one_num+=int(temp[-1].strip('\n'))-int(temp[1])
                label_lens.append(int(temp[-1].strip('\n'))-int(temp[1]))
        print('0的个数是：{}, 1的个数是{}'.format(zero_num-one_num,one_num))
        print('最短的标记长度是{},最长的标记长度是{}'.format(min(label_lens),max(label_lens)))

        for dev_artcicle in os.listdir(self.dev_articles):
            f=open(os.path.join(self.dev_articles,dev_artcicle),'r',
                   encoding='utf-8').read()
            sentence+=f
            s.append(list(f))
        token.fit_on_texts(sentence)
        if wordembedding:
            model=gensim.models.Word2Vec(s,size=self.embedding_dim,min_count=1)
            model.save('charembedding')
        if self.use_word2vec:
            model = gensim.models.Word2Vec.load('charembedding')
            embedding_matrix = np.zeros(shape=(len(model.wv.vocab.items()) + 1, model.vector_size))
            for char, i in token.word_index.items():
                embedding_matrix[i] = model.wv[char]
            return token, embedding_matrix
        else:
            return token,None

    def sort_length(self):
        train_len_dict={i:len(open(os.path.join(self.train_articles,os.listdir(self.train_articles)[i]),
                                   'r',encoding='utf-8').read()) for i in self.train_index}
        val_len_dict = {i: len(open(os.path.join(self.train_articles, os.listdir(self.train_articles)[i]),
                                      'r', encoding='utf-8').read()) for i in self.val_index}
        self.train_len_dict=sorted(train_len_dict.items(),key=lambda item:item[1])
        self.val_len_dict=sorted(val_len_dict.items(),key=lambda item:item[1])
        # print(self.train_len_dict)
        # print(self.val_len_dict)
    def generator(self,is_train=True):
        """
        batch size=1 因为训练样本数量不多，没必要padding了
        :param is_train:
        :return:
        """
        dicts=self.train_len_dict if is_train else self.val_len_dict
        index=[i[0] for i in dicts][::-1]
        lens=[i[1] for i in dicts][::-1]
        start=0
        while True:
            start = (start + 1) % len(index)
            flag=False
            if start+self.batch_size<len(index):
                batch_index=index[start:start+self.batch_size]
                batch_lens=lens[start:start+self.batch_size]
            else:
                flag=True
                have=len(index[start:])
                rest=self.batch_size-have
                batch_index=index[start:]+rest*[index[-1]]
                batch_lens=lens[start:]+rest*[lens[-1]]
            max_batch_lens = max(batch_lens)
            inputs = np.zeros(shape=(self.batch_size, max_batch_lens))
            labels = np.zeros(shape=(self.batch_size, max_batch_lens,1))
            for i, one_index in enumerate(batch_index):
                f = open(os.path.join(self.train_articles, os.listdir(self.train_articles)[one_index]), 'r',
                         encoding='utf-8').read()
                for j, char in enumerate(f):
                    inputs[i, j] = self.SI_token.word_index.get(char, 0)
                f = open(os.path.join(self.SI_dir,
                                      os.listdir(self.train_articles)[one_index].replace('.txt',
                                                                                         '.task1-SI.labels')),
                         'r',
                         encoding='utf-8').readlines()
                for j in f:
                    temp = j.split('\t')
                    begin = int(temp[1])
                    end = int(temp[-1].strip('\n'))
                    labels[i, begin:end,0] = 1
            # labels = K.one_hot(labels, 2)
            # print(labels.shape)
            yield inputs, labels
            start = (start + self.batch_size) % len(index) if flag==False else 0

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
        # crf = CRF()
        if use_crf:
            pass
            # x=Dense(self.hidden,activation='relu')(x)
            # x=Dense(1,activation='sigmoid')(x)
            # x =crf(x)
        else:
            x=Dense(self.hidden,activation='relu')(x)
            x=Dense(1,activation='sigmoid')(x)
        model=Model(input_layer,x)
        if use_crf == False:
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
        else:
            pass
            # model.compile(optimizer=Adam(0.001),loss=crf.loss,metrics=[crf.accuracy])
        model.summary()
        return model

def train(use_crf=False,use_word2vec=True):
    model_name='use_word2vec_bilstmcrf-depth_1-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5' if use_word2vec else 'bilstmcrf-depth_1-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5'
    data=Dataloader(use_word2vec=use_word2vec)
    model=Mymodel(vocab_size=data.char_num,hidden=256,dropout_rate=0.3,depth=1,embedding_dim=data.embedding_dim,
                  use_word2vec=use_word2vec,
                  embedding_matrxi=None if use_word2vec==False else data.embedding_matrix
                  ).build_network(use_crf=use_crf)

    model.fit_generator(
        generator=data.generator(is_train=True),
        steps_per_epoch=data.steps_per_epoch,
        validation_data=data.generator(is_train=False),
        validation_steps=data.val_steps_per_epoch,
        verbose=1,
        initial_epoch=0,
        epochs=100,
        class_weight=[1,6.5] if use_crf==False else None,
        callbacks=[
            TensorBoard('logs'),
            ReduceLROnPlateau(monitor='val_loss',patience=7,verbose=1),
            EarlyStopping(monitor='val_loss',patience=40,verbose=1,restore_best_weights=True),
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
train(use_crf=False,use_word2vec=False)

