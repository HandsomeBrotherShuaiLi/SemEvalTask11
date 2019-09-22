from keras import Input,Model
from keras.layers import LSTM,Bidirectional,Embedding,Dropout,Dense
from CRFlayer import CRF
from keras.optimizers import Adam
from Data import Dataloader
import os
from keras.callbacks import ReduceLROnPlateau,TensorBoard,ModelCheckpoint,EarlyStopping
class Mymodel(object):
    def __init__(self,vocab_size,hidden,dropout_rate,depth=1):
        self.vocab_size=vocab_size
        self.hidden=hidden
        self.dropout_rate=dropout_rate
        self.depth=depth

    def build_network(self,use_crf=False):
        input_layer=Input(shape=(None,),name='inputs')
        x=Embedding(self.vocab_size,self.hidden)(input_layer)
        x=Dropout(self.dropout_rate)(x)
        for i in range(self.depth):
            x = Bidirectional(LSTM(self.hidden, return_sequences=True),merge_mode='concat')(x)
        CRF_layer=CRF(units=1,learn_mode='join',test_mode='viterbi',sparse_target=False)
        if use_crf:
            x = CRF_layer(x)
        else:
            x=Dense(self.hidden,activation='relu')(x)
            x=Dense(1,activation='sigmoid')(x)
        model=Model(input_layer,x)
        model.summary()
        return model

def train(use_crf=False):
    model_name='bilstmcrf-depth_1-{epoch:03d}--{val_loss:.5f}--{val_acc:.5f}.hdf5'
    data=Dataloader()
    model=Mymodel(vocab_size=data.char_num,hidden=256,dropout_rate=0.3,depth=1).build_network(use_crf=use_crf)
    model.compile(optimizer=Adam(0.001),loss='binary_crossentropy',metrics=['acc'])
    model.fit_generator(
        generator=data.generator(is_train=True),
        steps_per_epoch=data.steps_per_epoch,
        validation_data=data.generator(is_train=False),
        validation_steps=data.val_steps_per_epoch,
        verbose=1,
        initial_epoch=0,
        epochs=100,
        class_weight=[1,6.5],
        callbacks=[
            TensorBoard('logs'),
            ReduceLROnPlateau(monitor='val_loss',patience=7,verbose=1),
            EarlyStopping(monitor='val_loss',patience=40,verbose=1,restore_best_weights=True),
            ModelCheckpoint(filepath=os.path.join('models',model_name),verbose=1,save_weights_only=False,
                            save_best_only=True,period=3)
        ]
    )

train(use_crf=False)


