import numpy as np
import os,json
def test_for_words():
    p='V2/word_level/train_labels'
    le=[]
    for name in os.listdir(p):
        f=np.load(os.path.join(p,name))
        le.append(len(f))
    print(le)
    print(max(le))
def test_for_word_level_prediction():
    json_path='results/dev_lstm_Word-level_lazyadam_Fixed-length-512_No-embedding_val_f1_2_1.json'
    txt_path='results/dev_lstm_Word-level_lazyadam_Fixed-length-512_No-embedding_val_f1_2_1.txt'
    prediction=json.load(open(json_path,'r',encoding='utf-8'))
    for p in prediction:
        word_index=np.load(p)
        words=prediction[p]
        for word in words:
            print(p,word,word_index[word,:])

if __name__=='__main__':
    test_for_word_level_prediction()