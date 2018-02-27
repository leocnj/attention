from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from attention_keras import Attention, Position_Embedding

max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.models import Model
from keras.layers import *

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)
# embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
# O_seq = Attention(4,8)([embeddings,embeddings,embeddings])
# O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(embeddings)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

earlystop = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
print('Train...')
model.fit(x_train, y_train,
          validation_split=0.1,
          batch_size=batch_size,
          epochs=5,
          callbacks=[earlystop])

score, acc = earlystop.model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score: ', score)
print('Test Acc: ', acc)