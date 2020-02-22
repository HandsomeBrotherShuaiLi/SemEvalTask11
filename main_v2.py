import os,tqdm,numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
from keras.preprocessing.text import Tokenizer
import kashgari
from keras_radam import RAdam
from kashgari.tasks.labeling import BiLSTM_CRF_Model,BiGRU_CRF_Model,BiGRU_Model,BiLSTM_Model
from kashgari.embeddings import BERTEmbedding,GPT2Embedding,WordEmbedding
from tensorflow import keras
import json
train_dir='V2/datasets/train-articles'
dev_dir='V2/datasets/dev-articles'
test_dir='V2/test-articles'
label_dir='V2/datasets/train-labels-task1-span-identification'
"""
思路：先给原文本做字符级别的mask,再用bert训练，再训练的时候，考虑采用裁断512的方法
"""
def test_for_article():
    name=os.listdir(train_dir)
    print(len(name),len(os.listdir(label_dir)))
    sl=0
    for i in name:
        path=os.path.join(train_dir,i)
        f=open(path,'r',encoding='utf-8').read()
        label=open(os.path.join(label_dir,i.replace('.txt','.task1-SI.labels'))).readlines()
        print(i,len(f))
        if len(f)>sl:sl=len(f)
        for line in label:
            _,s,e=line.strip('\n').split('\t')
            print(f[int(s):int(e)])
        print('*'*100)
    print(sl)

class Dataloader(object):
    def __init__(self,train_dir=train_dir,label_dir=label_dir,
                 test_dir=test_dir,dev_dir=dev_dir,split_rate=0.2,batch_size=8):
        """
        字符级别的数据读取器
        :param train_dir:
        :param label_dir:
        :param test_dir:
        :param dev_dir:
        :param split_rate:
        :param batch_size:
        """

        self.train_dir=train_dir
        self.label_dir=label_dir
        self.test_dir=test_dir
        self.dev_dir=dev_dir
        self.split_rate=split_rate
        self.batch_size=batch_size

    def init_mask(self,bert_len=512):
        mask=[]
        vac=[]
        for name in tqdm.tqdm(os.listdir(self.train_dir)):
            path=os.path.join(self.train_dir,name)
            label_path=os.path.join(self.label_dir,name.replace('.txt','.task1-SI.labels'))
            f_len=len(open(path,'r',encoding='utf-8').read())
            vac.append(open(path,'r',encoding='utf-8').read())
            temp=[]
            j=0
            while j<f_len:
                if j+bert_len<f_len:
                    temp.append([(j,j+bert_len),path,label_path])
                else:
                    temp.append([(j,f_len),path,label_path])
                j=j+bert_len
            mask+=temp
        for name in os.listdir(self.dev_dir):
            vac.append(open(os.path.join(self.dev_dir,name),'r',encoding='utf-8').read())
        for name in os.listdir(self.test_dir):
            vac.append(open(os.path.join(self.test_dir,name),'r',encoding='utf-8').read())
        token=Tokenizer(char_level=True,lower=False)
        token.fit_on_texts(vac)
        return mask,token
    def get_data(self,word_level=False,sentence_length=None):
        if word_level==False:
            #char 级别的处理
            if sentence_length is None:
                #整个文章全部送入模型
                train_name=np.array(os.listdir(self.train_dir))
                train_x,train_y,test_x,dev_x,val_x,val_y=[],[],[],[],[],[]
                all_index=np.array(range(len(train_name)))
                val_num=int(self.split_rate*len(train_name))
                val_index=np.random.choice(all_index,size=val_num,replace=False)
                train_index=np.array([i for i in all_index if i not in val_index])
                train_names=train_name[train_index]
                val_names=train_name[val_index]
                for i in train_names:
                    path=os.path.join(self.train_dir,i)
                    f=open(path,'r',encoding='utf-8').read()
                    train_x.append(list(f))
                    label_path=os.path.join(self.label_dir,i.replace('.txt','.task1-SI.labels'))
                    label=open(label_path,'r',encoding='utf-8').readlines()
                    mask=['0']*len(f)
                    for line in label:
                        line=line.strip('\n').split('\t')
                        mask[int(line[1]):int(line[2])]=['1']*(int(line[2])-int(line[1]))
                    train_y.append(mask)
                for i in val_names:
                    path=os.path.join(self.train_dir,i)
                    f=open(path,'r',encoding='utf-8').read()
                    val_x.append(list(f))
                    label_path=os.path.join(self.label_dir,i.replace('.txt','.task1-SI.labels'))
                    label=open(label_path,'r',encoding='utf-8').readlines()
                    mask=['0']*len(f)
                    for line in label:
                        line=line.strip('\n').split('\t')
                        mask[int(line[1]):int(line[2])]=['1']*(int(line[2])-int(line[1]))
                    val_y.append(mask)
                for i in os.listdir(self.test_dir):
                    path=os.path.join(self.test_dir,i)
                    f=open(path,'r',encoding='utf-8').read()
                    test_x.append(list(f))
                for i in os.listdir(self.dev_dir):
                    path=os.path.join(self.dev_dir,i)
                    f=open(path,'r',encoding='utf-8').read()
                    dev_x.append(list(f))
                return train_x,train_y,val_x,val_y,test_x,dev_x


class SemEval(object):
    def __init__(self,train_dir=train_dir,label_dir=label_dir,
                 test_dir=test_dir,dev_dir=dev_dir,batch_size=8,split_rate=0.1,
                 word_level=False,sentence_length=None):
        self.train_dir=train_dir
        self.label_dir=label_dir
        self.test_dir=test_dir
        self.dev_dir=dev_dir
        self.batch_size=batch_size
        self.split_rate=split_rate
        self.word_level=word_level
        self.sentence_length=sentence_length
        self.dataloader=Dataloader(train_dir=self.train_dir,label_dir=self.label_dir,
                                   test_dir=self.test_dir,dev_dir=self.dev_dir,
                                   batch_size=self.batch_size,split_rate=self.split_rate)
        self.train_x,self.train_y,self.val_x,self.val_y,self.test_x,\
        self.dev_x=self.dataloader.get_data(word_level,sentence_length)
    def build_model(self,model_name,embedding_name=None):
        emb_dict={
            'bert':BERTEmbedding(
                model_folder='/Users/shuai.li/Downloads/uncased_L-24_H-1024_A-16',
                sequence_length=512,
                trainable=True,
                task=kashgari.LABELING
            ),
        }

        if model_name.lower()=='lstm-crf':
            if embedding_name is None:
                model=BiLSTM_CRF_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiLSTM_CRF_Model(emb)
        elif model_name.lower()=='gru-crf':
            if embedding_name is None:
                model=BiGRU_CRF_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiGRU_CRF_Model(emb)
        elif model_name.lower()=='lstm':
            if embedding_name is None:
                model=BiLSTM_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiLSTM_Model(emb)
        else:
            if embedding_name is None:
                model=BiGRU_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiGRU_Model(emb)
        return model

    def train(self,model_name,embedding_name=None):
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        Model=self.build_model(model_name,embedding_name)
        Model.build_model(x_train=self.train_x,y_train=self.train_y,
                          x_validate=self.val_x,y_validate=self.val_y)
        opt=RAdam()
        emb_str= 'No-embedding' if embedding_name is None else embedding_name
        word_str='Word-level' if self.word_level else 'Char-level'
        sentence='Var-length' if self.sentence_length is None else 'Fixed-length'
        model_save_file='_'.join([model_name,word_str,sentence,emb_str])+'.h5'
        Model.compile_model(optimizer=opt)
        his=Model.fit(x_train=self.train_x,y_train=self.train_y,
                  x_validate=self.val_x,y_validate=self.val_y,batch_size=8,epochs=200,
                  callbacks=[
                      keras.callbacks.ModelCheckpoint(os.path.join('saved_models',model_save_file),monitor='val_loss',
                                                      verbose=1,save_weights_only=False,save_best_only=True),
                      keras.callbacks.ReduceLROnPlateau(patience=6,verbose=1,monitor='val_loss'),
                      keras.callbacks.TensorBoard('logs'),
                      keras.callbacks.EarlyStopping(monitor='val_loss',patience=40,verbose=1)
                  ])
        json.dump(his.history,open('saved_models/{}.json'.format(model_save_file.strip('.h5')),'w',encoding='utf-8'))

if __name__=='__main__':
    app=SemEval(word_level=False,sentence_length=None)
    app.train(model_name='lstm-crf',embedding_name=None)