import numpy as np
import os
def test_for_words():
    p='V2/word_level/train_labels'
    le=[]
    for name in os.listdir(p):
        f=np.load(os.path.join(p,name))
        le.append(len(f))
    print(le)
    print(max(le))


if __name__=='__main__':
    test_for_words()