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
from kashgari.corpus import CONLL2003ENCorpus
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
        self.mask,self.token=self.init_mask(bert_len=512)
        all_index=np.array(range(len(self.mask)))
        val_num=int(self.split_rate*len(self.mask))
        self.val_index=np.random.choice(all_index,size=val_num,replace=False)
        self.train_index=np.array([i for i in all_index if i not in self.val_index])
        self.train_steps=len(self.train_index)//self.batch_size
        self.val_steps=len(self.val_index)//self.batch_size
        print(self.token.word_index)
        print('vocab number:{}\ntrain number:{}\nval number:{}'.format(len(self.token.word_index),len(self.train_index),
                                                                       len(self.val_index)))
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
    def generator(self,is_train=True):
        index=self.train_index if is_train else self.val_index
        start=0
        while True:
            pass

class SemEval(object):
    def __init__(self,train_dir=train_dir,label_dir=label_dir,
                 test_dir=test_dir,dev_dir=dev_dir,batch_size=8,split_rate=0.2):
        self.train_dir=train_dir
        self.label_dir=label_dir
        self.test_dir=test_dir
        self.dev_dir=dev_dir
        self.batch_size=batch_size
        self.split_rate=split_rate
        self.dataloader=Dataloader(train_dir=self.train_dir,label_dir=self.label_dir,
                                   test_dir=self.test_dir,dev_dir=self.dev_dir,
                                   batch_size=self.batch_size,split_rate=self.split_rate)
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
            if embedding_name==None:
                model=BiLSTM_CRF_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiLSTM_CRF_Model(emb)
        elif model_name.lower()=='gru-crf':
            if embedding_name==None:
                model=BiGRU_CRF_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiGRU_CRF_Model(emb)
        elif model_name.lower()=='lstm':
            if embedding_name==None:
                model=BiLSTM_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiLSTM_Model(emb)
        else:
            if embedding_name==None:
                model=BiGRU_Model()
            else:
                emb=emb_dict[embedding_name.lower()]
                model=BiGRU_Model(emb)
        return model

    def train(self,model_name,embedding_name=None):
        model=self.build_model(model_name,embedding_name)
        model.fit()


if __name__=='__main__':
    train_x, train_y = CONLL2003ENCorpus.load_data('train')
    valid_x, valid_y = CONLL2003ENCorpus.load_data('valid')
    test_x, test_y = CONLL2003ENCorpus.load_data('test')
    print(train_x[0],train_x[1])
    print(train_y[0],train_y[1])