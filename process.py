import os,numpy as np
class Process(object):
    def __init__(self,train='data/train-articles',dev='data/dev-articles',
                 SI_labels='data/train-labels-task1-span-identification'):
        self.train=train
        self.dev=dev
        self.SI_labels=SI_labels
    def word_level_process(self):
        import string
        from zhon.hanzi import punctuation as p1
        from zhon.pinyin import punctuation as p2
        p=string.punctuation
        if not os.path.exists('data/train-word-level'):
            os.mkdir('data/train-word-level')
        if not os.path.exists('data/train-word-level-label'):
            os.mkdir('data/train-word-level-label')
        if not os.path.exists('data/dev-word-level'):
            os.mkdir('data/dev-word-level')
        if not os.path.exists('data/dev-word-index'):
            os.mkdir('data/dev-word-index')
        # for i in os.listdir(self.train):
        #     article = open(os.path.join(self.train, i), 'r', encoding='utf-8').read()
        #     labels=open(os.path.join(self.SI_labels,i.replace('.txt','.task1-SI.labels')),'r',encoding='utf-8').readlines()
        #     fragments=[[int(temp.split('\t')[1]),int(temp.split('\t')[-1].strip('\n'))] for temp in labels]
        #     word_list=[]
        #     start_index=[]
        #     end_index=[]
        #     temp=str()
        #     for idx, char in enumerate(article):
        #         if char!=' 'and char!='\n' and char!='\t' and char not in p and char not in p1 and char not in p2 and char !='':
        #             temp+=char
        #         else:
        #             if len(temp)>=1:
        #                 word_list.append(temp)
        #                 start_index.append(idx - len(temp))
        #                 end_index.append(idx)
        #             temp=str()
        #     word_label=[0]*len(word_list)
        #     for j in range(len(word_list)):
        #         start,end=start_index[j],end_index[j]
        #         for x in fragments:
        #             if x[0]<=start and end<=x[1]:
        #                 word_label[j]=1
        #                 break
        #     with open(os.path.join('data/train-word-level',i),'w',encoding='utf-8') as f:
        #         f.write(' '.join(word_list))
        #     f.close()
        #     with open(os.path.join('data/train-word-level-label',i.replace('.txt','.task1-SI.labels')),'w',encoding='utf-8') as f:
        #         f.write(' '.join([str(z) for z in word_label]))
        #     f.close()
        #     print(i+' Done!!')
        for i in os.listdir(self.dev):
            article = open(os.path.join(self.dev, i), 'r', encoding='utf-8').read()
            word_list = []
            word_start_end_index=[]
            temp = str()
            for idx, char in enumerate(article):
                if char != ' ' and char != '\n' and char != '\t' and char not in p and char not in p1 and char not in p2 and char != '':
                    temp += char
                else:
                    if len(temp) >= 1:
                        word_list.append(temp)
                        word_start_end_index.append([temp,idx - len(temp),idx])
                        # start_index.append(idx - len(temp))
                        # end_index.append(idx)
                    temp = str()
            with open(os.path.join('data/dev-word-level',i),'w',encoding='utf-8') as f:
                f.write(' '.join(word_list))
            f.close()
            word_start_end_index=np.array(word_start_end_index)
            np.save(os.path.join('data/dev-word-index',i.replace('.txt','_word_start_end_index.npy')),word_start_end_index)
def test_1():
    dev_dir='data/dev-articles'
    dev_word_level='data/dev-word-level'
    dev_index_dir='data/dev-word-index'
    for i in range(len(os.listdir(dev_dir))):
        raw_name=os.listdir(dev_dir)[i]
        word_level_name=os.listdir(dev_word_level)[i]
        index_file_name=os.listdir(dev_index_dir)[i]
        print(raw_name,word_level_name,index_file_name)
        raw_f=open(os.path.join(dev_dir,raw_name),'r',encoding='utf-8').read()
        word_level_f=open(os.path.join(dev_word_level,word_level_name),'r',encoding='utf-8').read().split(' ')
        word_index=np.load(os.path.join(dev_index_dir,index_file_name))
        for idx,word in enumerate(word_level_f):
            print(word)
            print(word_index[idx])
            print(raw_f[word_index[idx][0]:word_index[idx][1]])
            print('*'*100)

def test_2():
    """
    为什么一直单词级别的坐标和原文坐标对不上！！！！
    :return:
    """
    prediction=open('use_wordembedding_matrix--bilstm--depth_1--007--0.37131--0.84742_probs.txt',
                    'r',encoding='utf-8').readlines()
    for line in prediction:
        temp=line.strip('\n').split('\t')
        article_name=temp[0]
        prob=[float(i) for i in temp[1].split(' ')]
        raw_f=open('data/dev-articles/{}'.format(article_name),'r',encoding='utf-8').read()
        word_level_data=open('data/dev-word-level/{}'.format(article_name),'r',encoding='utf-8').read().split(' ')
        word_index=np.load('data/dev-word-index/{}'.format(article_name.replace('.txt','_word_start_end_index.npy')))
        temp=[]
        for i in range(len(prob)):
            if prob[i]>0.5:
                temp.append(i)
            else:
                if len(temp)>=1:
                    s,e=int(word_index[temp[0]][1]),int(word_index[temp[-1]][2])
                    print('根据word level找出的单词组是{}'.format(word_level_data[temp[0]:temp[-1]+1]))
                    print('根据原文找出的单词组是：{}'.format(raw_f[s:e]))
                temp=list()


test_2()