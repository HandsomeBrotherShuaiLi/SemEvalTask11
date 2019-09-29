import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from keras import Input,Model
from keras.layers import LSTM,Bidirectional,Embedding,Dropout,Dense,TimeDistributed
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.callbacks import Callback
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def boolMap(arr):
    if arr > 0.5:
        return 1
    else:
        return 0

class f1score(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.file_path = filepath

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_val_f1 = 0
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("max f1:{}".format(max(self.val_f1s)))
        if _val_f1 > self.best_val_f1:
            self.model.save_weights(self.file_path, overwrite=True)
            self.best_val_f1 = _val_f1
            print("best f1: {}".format(self.best_val_f1))
        else:
            print("val f1: {}, but not the best f1".format(_val_f1))
        return

class Wordlevelloader(object):
    def __init__(self,
                 train_dir='data/train-word-level',
                 dev_dir='data/dev-word-level',
                 train_label_dir='data/train-word-level-label',
                 dev_index_dir='data/dev-word-index',
                 split_rate=0.2,
                 batch_size=10,
                 embedding_dim=100,
                 use_pretained_embedding=False,
                 pretained_embedding_model=None):
        """

        :param train_dir:
        :param dev_dir:
        :param train_label_dir:
        :param dev_index_dir:
        :param split_rate:
        :param batch_size:
        :param embedding_dim:
        :param use_pretained_embedding: 如果不使用的话，那么Embedding层的参数权重不一样
        :param pretained_embedding_model:
        """
        self.train_dir=train_dir
        self.dev_dir=dev_dir
        self.train_label_dir=train_label_dir
        self.dev_index_dir=dev_index_dir
        self.split_rate=split_rate
        self.batch_size=batch_size
        self.embedding_dim=embedding_dim
        self.use_pretained_embedding=use_pretained_embedding
        self.pretained_embedding_model=pretained_embedding_model
        all_index=np.array(range(len(os.listdir(self.train_dir))),dtype='int')
        val_num=int(self.split_rate*len(os.listdir(self.train_dir)))
        if not os.path.exists('history_index'):
            self.val_index = np.random.choice(all_index, size=val_num, replace=False)
            self.train_index = np.array([i for i in all_index if i not in self.val_index],dtype='int')
            os.mkdir('history_index')
            np.save('history_index/train_index.npy',self.train_index)
            np.save('history_index/val_index.npy',self.val_index)
        else:
            self.val_index=np.load('history_index/val_index.npy')
            self.train_index=np.load('history_index/train_index.npy')
        self.steps_per_epoch=len(self.train_index)//self.batch_size
        self.val_steps_per_epoch=len(self.val_index)//self.batch_size
        print('全部数据是{}条,训练数据{}条，验证数据{}条'.format(len(all_index),len(self.train_index),
                                                len(self.val_index)))
        self.embedding_matrix,self.token,self.post_dim=self.process()
        self.word_num=len(self.token.word_index)
        # if self.embedding_matrix:
        #     print('本次将会导入词嵌入向量！')
    def process(self):
        sentences=[]
        all_num=0
        one_num=0
        train_sentences=[]
        train_labels=[]
        test_sentences=[]
        for i in os.listdir(self.train_dir):
            f=open(os.path.join(self.train_dir,i),'r',encoding='utf-8').read()
            label=open(os.path.join(self.train_label_dir,i.replace('.txt','.task1-SI.labels')),'r',encoding='utf-8').read()
            all_num+=len(label.split(' '))
            for j in label.split(' '):
                if j=='1':
                    one_num+=1
            sentences.append(f.split(' '))
            train_labels.append([int(z) for z in label.split(' ')])
            train_sentences.append(f.split(' '))
        for i in os.listdir(self.dev_dir):
            f = open(os.path.join(self.dev_dir, i), 'r', encoding='utf-8').read()
            sentences.append(f.split())
            test_sentences.append(f.split(' '))
        self.zero_rate=(all_num-one_num)/all_num
        print('0率:{}'.format(self.zero_rate))
        token=Tokenizer(char_level=False,lower=False)
        token.fit_on_texts(sentences)
        self.train_sequences=token.texts_to_sequences(train_sentences)#训练+交叉验证的数据
        self.train_labels=train_labels
        self.test_sequences=token.texts_to_sequences(test_sentences)#要预测的数据
        if self.use_pretained_embedding:
            if self.pretained_embedding_model==None:
                """
                自己用word2vec训练
                """
                model = gensim.models.Word2Vec(sentences, size=self.embedding_dim, min_count=1)
                embedding_matrix = np.zeros(shape=(len(token.word_index) + 1, self.embedding_dim))
                for word, i in token.word_index.items():
                    embedding_matrix[i] = model.wv[word]
                model.save('models/word_embedding')
                return embedding_matrix,token,self.embedding_dim
            else:
                if self.pretained_embedding_model=='models/word_embedding':
                    """
                    导入训练自己训练好的
                    """
                    model = gensim.models.Word2Vec.load(self.pretained_embedding_model)
                    embedding_matrix = np.zeros(shape=(len(token.word_index) + 1, model.vector_size))
                    for word, i in token.word_index.items():
                        embedding_matrix[i] = model.wv[word]
                    return embedding_matrix,token,model.vector_size
                else:
                    """
                    导入预训练好的
                    """
                    embeddings_index = {}
                    f = open(self.pretained_embedding_model, 'r', encoding='utf8')
                    dim = 0
                    count = 0
                    for line in f:
                        values = line.split()
                        word = ''.join(values[:-300])
                        coefs = np.asarray(values[-300:], dtype='float32')
                        embeddings_index[word] = coefs
                        dim = len(coefs)
                        embeddings_index[word] = coefs
                        count += 1
                        print('导入第{}个词：{}'.format(count, word))
                    f.close()
                    embedding_matrix = np.zeros(shape=(len(token.word_index) + 1, dim))
                    for word,i in token.word_index.items():
                        temp=embeddings_index.get(word)
                        if temp is not None:
                            embedding_matrix[i]=temp
                        else:
                            temp=embeddings_index.get(word.lower())
                            if temp is not None:
                                embedding_matrix[i] = temp
                            else:
                                z=word[0].upper()+word[1:].lower()
                                temp = embeddings_index.get(z, np.zeros(shape=(300,)))
                                embedding_matrix[i]=temp
                    return embedding_matrix,token,dim
        return None,token,self.embedding_dim

    def generator(self,is_train=True):
        index=self.train_index if is_train else self.val_index
        np.random.shuffle(index)
        data=np.array(self.train_sequences)
        start = 0
        while True:
            if start+self.batch_size<len(index):
                batch_index=index[start:(start+self.batch_size)]
            else:
                batch_index=np.hstack((index[start:],index[:(start+self.batch_size)%len(index)]))
            batch_inputs=pad_sequences(data[batch_index])
            batch_labels=pad_sequences(np.array(self.train_labels)[batch_index])
            batch_labels=np.expand_dims(batch_labels,axis=-1)
            yield batch_inputs,batch_labels
            start=(start+self.batch_size)%len(index)
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
        if self.use_wordvec==False:
            x = Embedding(self.vocab_size, self.embedding_dim)(input_layer)
        else:
            if self.embedding_matrix is not None:
                x = Embedding(self.vocab_size + 1, self.embedding_dim, weights=[self.embedding_matrix],
                              trainable=False)(
                    input_layer)
            else:
                x = Embedding(self.vocab_size, self.embedding_dim)(input_layer)
        x=Dropout(self.dropout_rate)(x)
        for i in range(self.depth):
            x = Bidirectional(LSTM(self.hidden, return_sequences=True),merge_mode='concat')(x)
        if use_crf:
            pass
        else:
            x=Dense(1024,activation='relu')(x)
            x=Dropout(self.dropout_rate)(x)
            x=Dense(1024,activation='relu')(x)
            x=TimeDistributed(Dense(1,activation='sigmoid'))(x)
        model=Model(input_layer,x)
        if use_crf == False:
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=[f1,'acc'])
        else:
            pass
        model.summary()
        return model

class Task11(object):
    def __init__(self,
                 train_dir='data/train-word-level',
                 dev_dir='data/dev-word-level',
                 train_label_dir='data/train-word-level-label',
                 dev_index_dir='data/dev-word-index',
                 split_rate=0.2,
                 batch_size=10,
                 embedding_dim=300,
                 use_pretained_embedding=False,
                 pretained_embedding_model=None
                 ):
        self.train_dir = train_dir
        self.dev_dir = dev_dir
        self.train_label_dir = train_label_dir
        self.dev_index_dir = dev_index_dir
        self.split_rate = split_rate
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.use_pretained_embedding = use_pretained_embedding
        self.pretained_embedding_model = pretained_embedding_model
    def train(self,use_crf=False,hidden=512,dropout=0.1,depth=1,predict=False):
        model_att=[]
        if self.use_pretained_embedding:
            model_att.append('use_wordembedding_matrix')
        else:
            pass
        if use_crf:
            model_att.append('crf-bilstm')
        else:
            model_att.append('bilstm')
        model_att.append('depth:{}'.format(str(depth)))
        model_full_name='--'.join(model_att)+'--{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5'
        data=Wordlevelloader(
            train_dir=self.train_dir,
            dev_dir=self.dev_dir,
            train_label_dir=self.train_label_dir,
            dev_index_dir=self.dev_index_dir,
            split_rate=self.split_rate,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            use_pretained_embedding=self.use_pretained_embedding,
            pretained_embedding_model=self.pretained_embedding_model
        )
        model=Mymodel(
            vocab_size=data.word_num,
            hidden=hidden,
            dropout_rate=dropout,
            use_word2vec=self.use_pretained_embedding,
            depth=depth,
            embedding_matrxi=data.embedding_matrix,
            embedding_dim=data.post_dim
        ).build_network(use_crf=use_crf)
        if predict==False:
            model.fit_generator(
                generator=data.generator(is_train=True),
                steps_per_epoch=data.steps_per_epoch,
                validation_data=data.generator(is_train=False),
                validation_steps=data.val_steps_per_epoch,
                verbose=1,
                initial_epoch=0,
                epochs=100,
                class_weight=[1 - data.zero_rate, data.zero_rate] if use_crf == False else None,
                callbacks=[
                    TensorBoard('logs'),
                    ReduceLROnPlateau(monitor='val_f1', patience=5, verbose=1, mode='max'),
                    EarlyStopping(monitor='val_f1', patience=28, verbose=1, restore_best_weights=True, mode='max'),
                    ModelCheckpoint(filepath=os.path.join('models', model_full_name), verbose=1,
                                    save_weights_only=False,
                                    save_best_only=True, monitor='val_f1', mode='max'),
                    # f1score(filepath=os.path.join('models', model_full_name))
                ]
            )
        else:
            return data,model
    def predict(self,model_path,use_crf=False,hidden=512,dropout=0.1,depth=1):
        data,model=self.train(use_crf=use_crf,hidden=hidden,dropout=dropout,depth=depth,
                              predict=True)
        print('开始导入模型...')
        model.load_weights(model_path)
        print('导入模型成功！')
        txt_name=model_path.split('/')[-1].strip('.hdf5')
        all_res = str()
        result_strict = open('submissions/{}_strict_submission.txt'.format(txt_name), 'w',
                             encoding='utf-8')
        result_soft= open('submissions/{}_soft_submission.txt'.format(txt_name), 'w',
                                encoding='utf-8')
        result_mid = open('submissions/{}_mid_submission.txt'.format(txt_name), 'w',
                           encoding='utf-8')
        result_strict_soft_merge = open('submissions/{}_strict_soft_merge_submission.txt'.format(txt_name), 'w',
                          encoding='utf-8')
        for idx,i in enumerate(os.listdir(data.dev_index_dir)):
            article_start_end=np.load(os.path.join(data.dev_index_dir,i))
            test_article=np.expand_dims(data.test_sequences[idx],axis=0)
            pred=model.predict(test_article)
            result_id=i.strip('article').strip('_word_start_end_index.npy')
            prob=pred[0:,:,0]
            dicts = {w: prob[w] for w in range(len(prob))}
            dict_sorted = sorted(dicts.items(), key=lambda item: item[1])
            zero_num = int(data.zero_rate * len(prob)) + 1
            mask = [0] * len(prob)
            for w in range(zero_num, len(prob)):
                mask[dict_sorted[w][0]] = 1
            #第一种预测，严格按照大于0.5来做
            print(result_id,min(prob), max(prob))
            if max(prob)>0.5:
                temp = list()
                for p in range(len(prob)):
                    if prob[p] > 0.5:
                        temp.append(p)
                    else:
                        if len(temp) >= 1:
                            all_res += result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                                article_start_end[temp[-1]][1]) + '\n'
                            result_strict.write(result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                                article_start_end[temp[-1]][1]) + '\n')
                            zz = data.test_sequences[idx][temp[0]:temp[-1]]
                            print('strict-mode中选择的句子:',data.token.sequences_to_texts(zz))
                        temp = list()
            print('\033[1;31;43m{}\033[0m'.format(result_id+' strict-mode Done!!!'))
            #第二种预测，按照0单词和1单词的比例分布,从低到高排列，选择最后比例个数的单词作为1
            temp = list()
            for w in range(len(mask)):
                if mask[w] == 1:
                    temp.append(w)
                else:
                    if len(temp)>=1:
                        all_res += result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                            article_start_end[temp[-1]][1]) + '\n'
                        result_soft.write(result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                            article_start_end[temp[-1]][1]) + '\n')
                        zz = data.test_sequences[idx][temp[0]:temp[-1]]
                        print('soft-mode中选择的句子:', data.token.sequences_to_texts(zz))
                    temp=list()
            print('\033[1;31;40m{}\033[0m'.format(result_id + ' soft-mode Done!!!'))
            #第三种预测：当最大的概率大于0.5时，就用strict模式，否则就是soft模式
            if max(prob) > 0.5:
                temp=list()
                for p in range(len(prob)):
                    if prob[p] > 0.5:
                        temp.append(p)
                    else:
                        if len(temp) >= 1:
                            all_res += result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                                article_start_end[temp[-1]][1]) + '\n'
                            result_mid.write(result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                                article_start_end[temp[-1]][1]) + '\n')
                            zz = data.test_sequences[idx][temp[0]:temp[-1]]
                            print('mid-mode中选择的句子:', data.token.sequences_to_texts(zz))
                        temp = list()
            else:
                temp = list()
                for w in range(len(mask)):
                    if mask[w] == 1:
                        temp.append(w)
                    else:
                        if len(temp) >= 1:
                            all_res += result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                                article_start_end[temp[-1]][1]) + '\n'
                            result_mid.write(result_id + '\t' + str(article_start_end[temp[0]][0]) + '\t' + str(
                                article_start_end[temp[-1]][1]) + '\n')
                            zz = data.test_sequences[idx][temp[0]:temp[-1]]
                            print('mid-mode中选择的句子:', data.token.sequences_to_texts(zz))
                        temp = list()
            print('\033[1;33;40m{}\033[0m'.format(result_id + ' mid-mode Done!!!'))
        result_strict.close()
        result_soft.close()
        result_mid.close()
        result_strict_soft_merge.write(all_res)
        result_strict_soft_merge.close()
if __name__=='__main__':
    #'F:\glove.840B.300d\glove.840B.300d.txt'
    #'./glove/glove.840B.300d.txt'
    t=Task11(
        batch_size=1,embedding_dim=300,use_pretained_embedding=True,
        pretained_embedding_model='F:\glove.6B\glove.6B.300d.txt'
    )
    t.predict(model_path='models/use_wordembedding_matrix--bilstm--depth_1--007--0.37131--0.84742.hdf5',
              use_crf=False,depth=1)
    # t.train(use_crf=False,hidden=512,dropout=0.1,depth=1)