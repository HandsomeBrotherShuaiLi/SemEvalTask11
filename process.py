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
        for i in os.listdir(self.train):
            article = open(os.path.join(self.train, i), 'r', encoding='utf-8').read()
            labels=open(os.path.join(self.SI_labels,i.replace('.txt','.task1-SI.labels')),'r',encoding='utf-8').readlines()
            fragments=[[int(temp.split('\t')[1]),int(temp.split('\t')[-1].strip('\n'))] for temp in labels]
            word_list=[]
            start_index=[]
            end_index=[]
            temp=str()
            for idx, char in enumerate(article):
                if char!=' 'and char!='\n' and char!='\t' and char not in p and char not in p1 and char not in p2 and char !='':
                    temp+=char
                else:
                    if len(temp)>=1:
                        word_list.append(temp)
                        start_index.append(idx - len(temp))
                        end_index.append(idx)
                    temp=str()
            word_label=[0]*len(word_list)
            for j in range(len(word_list)):
                start,end=start_index[j],end_index[j]
                for x in fragments:
                    if x[0]<=start and end<=x[1]:
                        word_label[j]=1
                        break
            with open(os.path.join('data/train-word-level',i),'w',encoding='utf-8') as f:
                f.write(' '.join(word_list))
            f.close()
            with open(os.path.join('data/train-word-level-label',i.replace('.txt','.task1-SI.labels')),'w',encoding='utf-8') as f:
                f.write(' '.join([str(z) for z in word_label]))
            f.close()
            print(i+' Done!!')
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
                        word_start_end_index.append([idx - len(temp),idx])
                        # start_index.append(idx - len(temp))
                        # end_index.append(idx)
                    temp = str()
            with open(os.path.join('data/dev-word-level',i),'w',encoding='utf-8') as f:
                f.write(' '.join(word_list))
            f.close()
            word_start_end_index=np.array(word_start_end_index)
            np.save(os.path.join('data/dev-word-index',i.replace('.txt','_word_start_end_index.npy')),word_start_end_index)
p=Process()
p.word_level_process()