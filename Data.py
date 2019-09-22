import os
from keras.preprocessing.text import Tokenizer
import numpy as np
import keras.backend as K
class Dataloader(object):
    """
    字符级别的处理，单词级别的处理很难，暂时不做
    """
    def __init__(self,train_articles='data/train-articles',
                 dev_articles='data/dev-articles',
                 SI_dir='data/train-labels-task1-span-identification',
                 TC_dir='data/train-labels-task2-technique-classification',
                 split_ratio=0.2,
                 batch_size=3):
        self.train_articles = train_articles
        self.dev_articles=dev_articles
        self.SI_dir=SI_dir
        self.TC_dir=TC_dir
        self.batch_size=batch_size
        self.SI_token=self.preprocess_for_SI()
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

    def preprocess_for_SI(self):
        token=Tokenizer(num_words=40000,char_level=True)
        sentence=[]
        zero_num=0
        one_num=0
        label_lens=[]
        for article in os.listdir(self.train_articles):
            path=os.path.join(self.train_articles,article)
            f=open(path,'r',encoding='utf-8').read()
            sentence+=f
            zero_num+=len(f)
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
        token.fit_on_texts(sentence)
        return token

    def sort_length(self):
        train_len_dict={i:len(open(os.path.join(self.train_articles,os.listdir(self.train_articles)[i]),
                                   'r',encoding='utf-8').read()) for i in self.train_index}
        val_len_dict = {i: len(open(os.path.join(self.train_articles, os.listdir(self.train_articles)[i]),
                                      'r', encoding='utf-8').read()) for i in self.val_index}
        self.train_len_dict=sorted(train_len_dict.items(),key=lambda item:item[1])
        self.val_len_dict=sorted(val_len_dict.items(),key=lambda item:item[1])
    def generator(self,is_train=True):
        """
        batch size=1 因为训练样本数量不多，没必要padding了
        :param is_train:
        :return:
        """
        dicts=self.train_len_dict if is_train else self.val_len_dict
        index=[i[0] for i in dicts]
        lens=[i[1] for i in dicts]
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

