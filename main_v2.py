import os,tqdm,numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow
import tensorflow as tf
import codecs
from keras_bert import Tokenizer as Bert_Tokenizer
from keras_radam import RAdam
# from tensorflow.keras.backend import set_session
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tensorflow.Session(config=config))
from keras.preprocessing.text import Tokenizer
from models.Models import CustomModels
from collections import defaultdict
import json,string
from zhon.hanzi import punctuation as p1
from zhon.pinyin import punctuation as p2
train_dir='V2/datasets/train-articles'
dev_dir='V2/datasets/dev-articles'
test_dir='V2/test-articles'
label_dir='V2/datasets/train-labels-task1-span-identification'
"""
思路：
1. 字符级的BiLSTM
2. 词级别的BiLSTM
3. 词的embedding + BiLSTM
4  字符+词（embedding)+BiLSTM
"""
pretrained_models={
    'multi_cased_base':'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip',
    'chinese_base':'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip',
    'wwm_uncased_large' : 'https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip',
    'wwm_cased_large':'https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip'
}

class Dataloader(object):
    def __init__(self,train_dir=train_dir,label_dir=label_dir,
                 test_dir=test_dir,dev_dir=dev_dir,split_rate=0.2,batch_size=8,
                 fixed_length=512,word_level=False,mixed=False,embedding=None):
        """

        :param train_dir:
        :param label_dir:
        :param test_dir:
        :param dev_dir:
        :param split_rate:
        :param batch_size:
        :param fixed_length:
        :param word_level:
        :param mixed:
        """

        self.train_dir=train_dir
        self.label_dir=label_dir
        self.test_dir=test_dir
        self.dev_dir=dev_dir
        self.split_rate=split_rate
        self.batch_size=batch_size
        self.fixed_length=fixed_length
        self.word_level=word_level
        self.mixed=mixed
        self.embedding=embedding
        self.mask,self.token,self.dev,self.test,self.paths=self.init_mask(self.fixed_length,self.word_level,self.embedding)
        all_index=np.array(range(len(self.mask)))
        val_num=int(self.split_rate*len(self.mask))
        self.val_index=np.random.choice(all_index,size=val_num,replace=False)
        self.train_index=np.array([i for i in all_index if i not in self.val_index])
        self.train_steps=len(self.train_index)//self.batch_size
        self.val_steps=len(self.val_index)//self.batch_size
        vocab_len=len(self.token.word_index) if self.embedding is None else len(self.token)
        print('vocab number:{}\ntrain number:{}\nval number:{}\ndev number:{}\ntest number:{}'.format(vocab_len,len(self.train_index),
                                                                       len(self.val_index),len(self.dev),len(self.test)))

    def download(self,kind,dir='pretrained_embedding'):
        url=pretrained_models[kind]
        if not os.path.exists(dir):
            os.mkdir(dir)
        fname=url.split('/')[-1].strip('.zip')
        p=os.getcwd()
        # print('start to wget......')
        # os.system('wget -P {} {}'.format(os.path.join(p,dir),url))
        # print('start to unzip......')
        # os.system('unzip -d {} {}'.format(os.path.join(p,dir),os.path.join(p,dir,url.split('/')[-1])))
        # os.system('rm {}'.format(os.path.join(p,dir,url.split('/')[-1])))
        config_path=os.path.join(p,dir,fname,'bert_config.json')
        checkpoints_path=os.path.join(p,dir,fname,'bert_model.ckpt')
        vocab_path=os.path.join(p,dir,fname,'vocab.txt')
        token_dict={}
        with codecs.open(vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return config_path,checkpoints_path,vocab_path,token_dict


    def init_mask(self,bert_len=512,word_level=False,embedding=None):
        """

        :param bert_len:
        :param word_level:
        :return:
        """
        if word_level==False:
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
                        temp.append([j,j+bert_len,path,label_path])
                    else:
                        temp.append([j,f_len,path,label_path])
                    j=j+bert_len
                mask+=temp
            dev=[]
            for name in os.listdir(self.dev_dir):
                f=open(os.path.join(self.dev_dir,name),'r',encoding='utf-8').read()
                vac.append(f)
                temp=[]
                j=0
                while j<len(f):
                    if j+bert_len<len(f):
                        temp.append([j,j+bert_len,os.path.join(self.dev_dir,name)])
                    else:
                        temp.append([j,len(f),os.path.join(self.dev_dir,name)])
                    j+=bert_len
                dev+=temp
            test=[]
            for name in os.listdir(self.test_dir):
                f=open(os.path.join(self.test_dir,name),'r',encoding='utf-8').read()
                vac.append(f)
                temp=[]
                j=0
                while j<len(f):
                    if j+bert_len<len(f):
                        temp.append([j,j+bert_len,os.path.join(self.test_dir,name)])
                    else:
                        temp.append([j,len(f),os.path.join(self.test_dir,name)])
                    j+=bert_len
                test+=temp
            token=Tokenizer(char_level=True,lower=False)
            token.fit_on_texts(vac)
            return mask,token,dev,test,None
        else:
            if not os.path.exists('V2/word_level'):
                os.mkdir('V2/word_level')
                os.mkdir('V2/word_level/train_articles')
                os.mkdir('V2/word_level/train_labels')
                os.mkdir('V2/word_level/dev')
                os.mkdir('V2/word_level/test')
            sentences=[]
            sentences+=self.char2word(self.train_dir,self.label_dir)
            sentences+=self.char2word(self.test_dir)
            sentences+=self.char2word(self.dev_dir)
            F_path=None
            if embedding is None:
                token=Tokenizer(char_level=False,lower=False)
                token.fit_on_texts(sentences)
            else:
                kind='wwm_cased_large'
                config_path,checkpoints_path,vocab_path,token_dict=self.download(kind=kind)
                token=token_dict
                F_path=[config_path,checkpoints_path,vocab_path]

            if bert_len is None:
                samples=[]
                dev=[]
                test=[]
                for name in tqdm.tqdm(os.listdir('V2/word_level/train_articles'),
                                      total=len(os.listdir('V2/word_level/train_articles'))):
                    samples.append([os.path.join('V2/word_level/train_articles',name),os.path.join('V2/word_level/train_labels',name)])
                for name in tqdm.tqdm(os.listdir('V2/word_level/dev'),total=len(os.listdir('V2/word_level/dev'))):
                    dev.append(os.path.join('V2/word_level/dev',name))
                for name in tqdm.tqdm(os.listdir('V2/word_level/test'),total=len(os.listdir('V2/word_level/test'))):
                    test.append(os.path.join('V2/word_level/test',name))
                return samples,token,dev,test,F_path
            else:
                samples=[]
                dev=[]
                test=[]
                for name in tqdm.tqdm(os.listdir('V2/word_level/train_articles'),
                                      total=len(os.listdir('V2/word_level/train_articles'))):
                    f=np.load(os.path.join('V2/word_level/train_articles',name))
                    word_number=len(f)
                    j=0
                    temp=[]
                    while j<word_number:
                        if j+bert_len<word_number:
                            temp.append([j,j+bert_len,
                                         os.path.join('V2/word_level/train_articles',name),
                                         os.path.join('V2/word_level/train_labels',name)])
                        else:
                            temp.append([
                                j,word_number,
                                os.path.join('V2/word_level/train_articles',name),
                                os.path.join('V2/word_level/train_labels',name)
                            ])
                        j+=bert_len
                    samples+=temp
                for name in tqdm.tqdm(os.listdir('V2/word_level/dev'),total=len(os.listdir('V2/word_level/dev'))):
                    f=np.load(os.path.join('V2/word_level/dev',name))
                    word_number=len(f)
                    j=0
                    temp=[]
                    while j<word_number:
                        if j+bert_len<word_number:
                            temp.append([
                                j,j+bert_len,
                                os.path.join('V2/word_level/dev',name)
                            ])
                        else:
                            temp.append(
                                [
                                    j,word_number,os.path.join('V2/word_level/dev',name)
                                ]
                            )
                        j+=bert_len
                    dev+=temp
                for name in tqdm.tqdm(os.listdir('V2/word_level/test'),total=len(os.listdir('V2/word_level/test'))):
                    f=np.load(os.path.join('V2/word_level/test',name))
                    word_number=len(f)
                    j=0
                    temp=[]
                    while j<word_number:
                        if j+bert_len<word_number:
                            temp.append(
                                [j,j+bert_len,os.path.join('V2/word_level/test',name)]
                            )
                        else:
                            temp.append(
                                [j,word_number,os.path.join('V2/word_level/test',name)]
                            )
                        j+=bert_len
                    test+=temp
                return samples,token,dev,test,F_path

    def char2word(self,article_path,label_dir=None):
        """

        :param article_path:
        :param label_dir:
        :return:
        """
        mapping={
            'train-articles':'V2/word_level/train_articles',
            'dev-articles':'V2/word_level/dev',
            'test-articles':'V2/word_level/test',
            'train-labels-task1-span-identification':'V2/word_level/train_labels'
        }
        sentences=[]
        for name in tqdm.tqdm(os.listdir(article_path),total=len(os.listdir(article_path))):
            path=os.path.join(article_path,name)
            if label_dir:
                label_path=os.path.join(label_dir,name.replace('.txt','.task1-SI.labels'))
                label=open(label_path,'r',encoding='utf-8').readlines()
                mask=[]
            f=open(path,'r',encoding='utf-8').read()
            word_index=[]
            p3=string.punctuation
            temp=str()
            for i,char in enumerate(f):
                if char not in [' ','\n','\t',''] and char not in p1 and char not in p2 and char not in p3:
                    temp+=char
                else:
                    if len(temp)>=1:
                        word_index.append([temp,i-len(temp),i])
                        sentences.append(temp)
                        if label_dir is not None:
                            flag=0
                            for line in label:
                                _,s,e=line.strip('\n').split('\t')
                                if i-len(temp)>=int(s)-1 and i<=int(e)+1:
                                    flag=1
                                    break
                            mask.append(flag)
                    temp=str()
            word_index=np.array(word_index)
            np.save(os.path.join(mapping[article_path.split('/')[-1]],name.replace('.txt','.npy')),word_index)
            if label_dir:
                np.save(os.path.join(mapping[label_dir.split('/')[-1]],name.replace('.txt','.npy')),np.array(mask))
        return sentences

    def get_data(self,word_level=False,sentence_length=None):
        """

        :param word_level:
        :param sentence_length:
        :return:
        """
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
                ls=[]
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
                    ls+=mask
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
                    ls+=mask
                    val_y.append(mask)
                for i in os.listdir(self.test_dir):
                    path=os.path.join(self.test_dir,i)
                    f=open(path,'r',encoding='utf-8').read()
                    test_x.append(list(f))
                for i in os.listdir(self.dev_dir):
                    path=os.path.join(self.dev_dir,i)
                    f=open(path,'r',encoding='utf-8').read()
                    dev_x.append(list(f))
                return train_x,train_y,val_x,val_y,test_x,dev_x,set(ls)

    def generator(self,is_train=True):
        """

        :param is_train:
        :return:
        """
        index=self.train_index if is_train else self.val_index
        start=0
        if self.word_level==False:
            while True:
                inputs=np.zeros(shape=(self.batch_size,self.fixed_length))
                labels=np.zeros(shape=(self.batch_size,self.fixed_length,1))
                if start+self.batch_size<len(index):
                    batch_index=index[start:start+self.batch_size]
                else:
                    batch_index=np.hstack((index[start:],index[:(start+self.batch_size)%len(index)]))
                np.random.shuffle(batch_index)
                for c,i in enumerate(batch_index):
                    f_i,f_j,path,label_path=self.mask[i]
                    file=open(path,'r',encoding='utf-8').read()[f_i:f_j]
                    text2id=np.array(self.token.texts_to_sequences(file))
                    text2id=np.squeeze(text2id,axis=-1)
                    if len(text2id)==self.fixed_length:
                        inputs[c,:]=text2id
                    else:
                        inputs[c,:len(text2id)]=text2id
                    for line in open(label_path,'r',encoding='utf-8').readlines():
                        _,s,e=line.strip('\n').split('\t')
                        if int(s)>=f_i and int(e)<=f_j:
                            labels[c,(int(s)-f_i):(int(e)-f_i),0]=1
                        elif int(s)>=f_i and int(e)>f_j:
                            labels[c,(int(s)-f_i):,0]=1
                        elif int(s)<f_i and int(e)<=f_j:
                            labels[c,:(int(e)-f_i),0]=1
                        else:
                            labels[c,:,0]=1
                yield inputs,labels
                start=(start+self.batch_size)%len(index)
        else:
            if self.fixed_length and self.embedding is None:
                while True:
                    inputs=np.zeros(shape=(self.batch_size,self.fixed_length))
                    labels=np.zeros(shape=(self.batch_size,self.fixed_length,1))
                    if start+self.batch_size<len(index):
                        batch_index=index[start:start+self.batch_size]
                    else:
                        batch_index=np.hstack((index[start:],index[:(start+self.batch_size)%len(index)]))
                    np.random.shuffle(index)
                    for c,i in enumerate(batch_index):
                        word_i,word_j,path,label_path=self.mask[i]
                        words=np.load(path)[word_i:word_j,0]
                        label=np.load(label_path)[word_i:word_j]
                        words2id=np.array(self.token.texts_to_sequences(words))
                        words2id=np.squeeze(words2id,axis=-1)
                        if len(words2id)!=len(label):
                            raise ValueError('词和标签数量不同')
                        if len(words2id)==self.fixed_length:
                            inputs[c,:]=words2id
                            labels[c,:,:]=np.expand_dims(label,axis=-1)
                        else:
                            inputs[c,:len(words2id)]=words2id
                            labels[c,:len(words2id),:]=np.expand_dims(label,axis=-1)
                    yield inputs,labels
                    start=(start+self.batch_size)%len(index)
            elif self.fixed_length and self.embedding is not None:
                bert_tokenizer=Bert_Tokenizer(self.token)
                while True:
                    input_tokens=np.zeros(shape=(self.batch_size,self.fixed_length))
                    input_segments=np.zeros(shape=(self.batch_size,self.fixed_length))
                    input_labels=np.zeros(shape=(self.batch_size,self.fixed_length,1))
                    if start+self.batch_size<len(index):
                        batch_index=index[start:start+self.batch_size]
                    else:
                        batch_index=np.hstack((index[start:],index[:(start+self.batch_size)%len(index)]))
                    np.random.shuffle(index)
                    for c,i in enumerate(batch_index):
                        word_i,word_j,path,label_path=self.mask[i]
                        words=np.load(path)[word_i:word_j,0]
                        words=' '.join(words)
                        label=np.load(label_path)[word_i:word_j]
                        ids,segments=bert_tokenizer.encode(words,max_len=self.fixed_length)
                        input_tokens[c,:]=ids
                        input_segments[c,:]=segments
                        if len(label)==self.fixed_length:
                            input_labels[c,:,:]=np.expand_dims(label,axis=-1)
                        else:
                            input_labels[c,:len(label),:]=np.expand_dims(label,axis=-1)
                    yield [input_tokens,input_segments],input_labels
                    start=(start+self.batch_size)%len(index)


class SemEval(object):
    def __init__(self,train_dir=train_dir,label_dir=label_dir,
                 test_dir=test_dir,dev_dir=dev_dir,batch_size=8,split_rate=0.1,
                 word_level=False,fixed_length=512,mixed=False,embedding=None):
        """

        :param train_dir:
        :param label_dir:
        :param test_dir:
        :param dev_dir:
        :param batch_size:
        :param split_rate:
        :param word_level:
        :param fixed_length:
        :param mixed:
        """
        self.train_dir=train_dir
        self.label_dir=label_dir
        self.test_dir=test_dir
        self.dev_dir=dev_dir
        self.batch_size=batch_size
        self.split_rate=split_rate
        self.word_level=word_level
        self.fixed_length=fixed_length
        self.mixed=mixed
        self.embedding=embedding
        self.dataloader=Dataloader(train_dir=self.train_dir,label_dir=self.label_dir,
                                   test_dir=self.test_dir,dev_dir=self.dev_dir,
                                   batch_size=self.batch_size,split_rate=self.split_rate,
                                   word_level=self.word_level,fixed_length=self.fixed_length,
                                   mixed=self.mixed,embedding=self.embedding)

    def train(self,model_name,monitor='val_acc',layer_number=2,lr=0.001,
              op_setting='adam',trainable=True):
        """

        :param model_name:
        :param embedding_name:
        :param monitor:
        :param layer_number:
        :param lr:
        :param op_setting:
        :return:
        """
        if not os.path.exists('saved_models'):
            os.mkdir('saved_models')
        modes={
            'val_acc':'max',
            'val_loss':'min',
            'val_f1':'max'
        }
        emb_str= 'No-embedding' if self.embedding is None else self.embedding
        word_str='Word-level' if self.word_level else 'Char-level'
        sentence='Var-length' if self.fixed_length is None else 'Fixed-length-{}'.format(self.fixed_length)
        v_s=len(self.dataloader.token.word_index)+1 if self.embedding is None else len(self.dataloader.token.keys())
        train_str='trainable' if trainable else 'freezed'
        model_save_file='_'.join([model_name,word_str,op_setting,sentence,train_str,emb_str,monitor,str(layer_number)])+'.h5'
        model,loss,metrics,embed_layer=CustomModels(model_name=model_name,vocab_size=v_s,
                           embedding_name=self.embedding,layer_number=layer_number,paths=self.dataloader.paths).build_model(trainable=trainable)
        if self.embedding is None:
            from tensorflow.keras.optimizers import Adam,RMSprop,SGD
            from models.CRF import f1,LazyOptimizer
            op_dict={
                'adam':Adam(lr),
                'radam':RAdam(lr),
                'rms':RMSprop(lr),
                'sgd':SGD(lr),
                'lazyadam':LazyOptimizer(Adam(lr),[embed_layer]),
                'lazyrms':LazyOptimizer(RMSprop(lr),[embed_layer])
            }
            callbacks=[
                tensorflow.keras.callbacks.ModelCheckpoint(os.path.join('saved_models',model_save_file),
                                                   monitor=monitor,verbose=1,save_best_only=True,
                                                   save_weights_only=False,mode=modes[monitor]),
                tensorflow.keras.callbacks.TensorBoard('logs'),
                tensorflow.keras.callbacks.EarlyStopping(monitor=monitor,patience=40,verbose=1,mode=modes[monitor]),
                tensorflow.keras.callbacks.ReduceLROnPlateau(monitor=monitor,patience=6,verbose=1,mode=modes[monitor])
            ]
            op=op_dict[op_setting.lower()]
            model.compile(optimizer=op,loss=loss,metrics=metrics+[f1])
        else:
            from keras.optimizers import Adam,RMSprop,SGD
            import keras
            from models.f1_keras import f1
            op_dict={
                'adam':Adam(lr),
                'radam':RAdam(lr),
                'rms':RMSprop(lr),
                'sgd':SGD(lr)
            }
            callbacks=[
                keras.callbacks.ModelCheckpoint(os.path.join('saved_models',model_save_file),
                                                           monitor=monitor,verbose=1,save_best_only=True,
                                                           save_weights_only=False,mode=modes[monitor]),
                keras.callbacks.TensorBoard('logs'),
                keras.callbacks.EarlyStopping(monitor=monitor,patience=40,verbose=1,mode=modes[monitor]),
                keras.callbacks.ReduceLROnPlateau(monitor=monitor,patience=6,verbose=1,mode=modes[monitor])

            ]
            op=op_dict[op_setting.lower()]
            model.compile(optimizer=op,loss=loss,metrics=metrics+[f1])

        model.fit_generator(
            generator=self.dataloader.generator(is_train=True),
            steps_per_epoch=self.dataloader.train_steps,
            validation_data=self.dataloader.generator(is_train=False),
            validation_steps=self.dataloader.val_steps,
            verbose=1,initial_epoch=0,epochs=200,
            callbacks=callbacks
        )
    def predict(self,model_path):
        """

        :param model_path:
        :return:
        """

        model_name=model_path.split('/')[-1].strip('.h5')
        direct_load=True if model_name.split('_')[0].endswith('crf')==False else False
        direct_load=False
        if direct_load==False:
            v_s=len(self.dataloader.token.word_index)+1 if self.embedding is None else len(self.dataloader.token.keys())
            e_n=model_name.split('_')[-4] if model_name.split('_')[-4]!='No-embedding' else None
            layer=int(model_name.split('_')[-1])
            model,loss,metrics,embedding_layer=CustomModels(model_name=model_name.split('_')[0],
                                            vocab_size=v_s,
                                            embedding_name=e_n,layer_number=layer,paths=self.dataloader.paths).build_model()
            model.load_weights(model_path)
        else:
            if self.embedding is None:
                from tensorflow.keras.models import load_model
                model=load_model(model_path,compile=False)
            else:
                from keras.models import load_model
                model=load_model(model_path,compile=False)
        if self.word_level==False:
            for g in range(1,51):
                self.helper_char_level(model_name=model_name,model=model,data='dev',gap=g)
                self.helper_char_level(model_name=model_name,model=model,data='test',gap=g)
        else:
            for g in range(1,51):
                self.helper_word_level(model_name=model_name,model=model,data='dev',gap=g)
                self.helper_word_level(model_name=model_name,model=model,data='test',gap=g)
    def helper_char_level(self,model,model_name,data:str,gap:int):
        """

        :param data:
        :return:
        """
        result=defaultdict(list)
        dataset={
            'dev':self.dataloader.dev,
            'test':self.dataloader.test
        }
        for line in tqdm.tqdm(dataset[data],total=len(dataset[data])):
            i,j,path=line
            f=open(path,'r',encoding='utf-8').read()[i:j]
            seg=np.array(self.dataloader.token.texts_to_sequences(f))
            seg=np.squeeze(seg,axis=-1)
            try:
                seg=np.expand_dims(seg,axis=0)
                pred=np.squeeze(model.predict(seg)[0],axis=-1)
            except:
                seg[np.where(seg>=len(self.dataloader.token.word_index))[0]]=0
                seg=np.expand_dims(seg,axis=0)
                pred=np.squeeze(model.predict(seg)[0],axis=-1)
            if model_name.split('_')[0].endswith('crf'):print(pred)
            pro_index=np.where(pred>=0.5)
            result[path]+=np.array(pro_index[0]+i).tolist()
        #json.dump(result,open('results/{}_{}_{}.json'.format(data,model_name,gap),'w',encoding='utf-8'))
        sub=open('results/{}_{}_{}.txt'.format(data,model_name,gap),'w',encoding='utf-8')
        for path in tqdm.tqdm(result,total=len(result)):
            all_index=result[path]
            id=path.split('/')[-1].strip('.txt').strip('article')
            if len(all_index)==0:
                pass
            elif len(all_index)==1:
                sub.write('{}\t{}\t{}\n'.format(id,all_index[0],all_index[0]+1))
            else:
                start=all_index[0]
                for c in range(1,len(all_index)):
                    if all_index[c]-all_index[c-1]<=gap:
                        pass
                    else:
                        end=all_index[c-1]+1
                        sub.write('{}\t{}\t{}\n'.format(id,start,end))
                        start=all_index[c]
                sub.write('{}\t{}\t{}\n'.format(id,start,all_index[-1]+1))
        sub.close()
    def helper_word_level(self,model,model_name,data:str,gap:int):
        result=defaultdict(list)
        dataset={
            'dev':self.dataloader.dev,
            'test':self.dataloader.test
        }
        bert_tokenizer=None
        if self.embedding:
            bert_tokenizer=Bert_Tokenizer(self.dataloader.token)
        if self.fixed_length is None:
            for line in tqdm.tqdm(dataset[data],total=len(dataset[data])):
                f=np.load(line)
                if self.embedding is None:
                    words=self.dataloader.token.texts_to_sequences(f[:,0])
                    words=np.squeeze(words,axis=-1)
                    words=np.expand_dims(words,axis=0)
                    pred=np.squeeze(model.predict(words)[0],axis=-1)
                    if model_name.split('_')[0].endswith('crf'):print(pred)
                    pro_index=np.where(pred>=0.5)
                    result[line]+=pro_index[0].to_list()
                else:
                    words=' '.join(f[:,0])
                    ids,segments=bert_tokenizer.encode(words)
                    ids=np.expand_dims(ids,axis=0)
                    segments=np.expand_dims(segments,axis=0)
                    pred=np.squeeze(model.predict([ids,segments])[0],axis=-1)
                    if model_name.split('_')[0].endswith('crf'):print(pred)
                    pro_index=np.where(pred>=0.5)
                    result[line]+=pro_index[0].to_list()

        else:
            for line in tqdm.tqdm(dataset[data],total=len(dataset[data])):
                s,e,path=line
                f=np.load(path)
                if self.embedding is None:
                    words=self.dataloader.token.texts_to_sequences(f[s:e,0])
                    words=np.squeeze(words,axis=-1)
                    words=np.expand_dims(words,axis=0)
                    pred=np.squeeze(model.predict(words)[0],axis=-1)
                    if model_name.split('_')[0].endswith('crf'):print(pred)
                    pro_index=np.where(pred>=0.5)
                    result[path]+=np.array(pro_index[0]+s).tolist()
                else:
                    words=' '.join(f[s:e,0])
                    ids,segments=bert_tokenizer.encode(words,max_len=self.fixed_length)
                    ids=np.expand_dims(ids,axis=0)
                    segments=np.expand_dims(segments,axis=0)
                    pred=np.squeeze(model.predict([ids,segments])[0],axis=-1)
                    if model_name.split('_')[0].endswith('crf'):print(pred)
                    pro_index=np.where(pred>=0.5)
                    result[line]+=pro_index[0].to_list()
        json.dump(result,open('results/{}_{}_{}.json'.format(data,model_name,gap),'w',encoding='utf-8'))
        sub=open('results/{}_{}_{}.txt'.format(data,model_name,gap),'w',encoding='utf-8')
        for path in tqdm.tqdm(result,total=len(result)):
            word_index=result[path]
            word_start_end=np.load(path)
            id=path.split('/')[-1].strip('.npy').strip('article')
            if len(word_index)==0:
                pass
            elif len(word_index)==1:
                sub.write('{}\t{}\t{}\n'.format(id,word_start_end[word_index[0],1],word_start_end[word_index[0],2]))
            else:
                start=word_index[0]
                for c in range(1,len(word_index)):
                    if word_index[c]-word_index[c-1]<=gap:
                        pass
                    else:
                        end=word_start_end[word_index[c-1],2]
                        sub.write('{}\t{}\t{}\n'.format(id,word_start_end[start,1],end))
                        start=word_index[c]
                sub.write('{}\t{}\t{}\n'.format(id,word_start_end[start,1],word_start_end[word_index[-1],2]))
        sub.close()
if __name__=='__main__':
    app=SemEval(
        split_rate=0.2,batch_size=2,word_level=True,embedding='bert',fixed_length=512
    )
    app.predict('saved_models/lstm_Word-level_adam_Fixed-length-512_freezed_bert_val_acc_2.h5')
    app.predict('saved_models/lstm_Word-level_adam_Fixed-length-512_freezed_bert_val_f1_2.h5')
    app.predict('saved_models/lstm_Word-level_adam_Fixed-length-512_trainable_bert_val_f1_2.h5')
    app.predict('saved_models/lstm_Word-level_adam_Fixed-length-512_trainable_bert_val_acc_2.h5')
