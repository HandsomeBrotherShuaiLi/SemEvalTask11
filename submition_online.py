import requests,time
def submit():
    url='https://propaganda.qcri.org/semeval2020-task11/teampage.php?passcode=c10fbc2a11a4b2a70ba211af31400c7a'
    file={'file':open('results/dev_lstm_Word-level_lazyadam_Fixed-length-512_No-embedding_val_f1_2_1.txt','rb')}
    data={
        'team':'LS','dataset':'dev'
    }
    response=requests.post(url,files=file,data=data)
    time.sleep(10)
    print(response.text)

if __name__=='__main__':
    submit()