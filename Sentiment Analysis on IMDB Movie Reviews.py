#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

_word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data= keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen=250)
test_data= keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary() 
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitmodel= model.fit(x_train,y_train,epochs=43,verbose=1,batch_size=512)
results = model.evaluate(test_data , test_labels)

model.save('my_ronit.keras')


model = keras.models.load_model("my_ronit.keras")
def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded

with open ('test.txt',encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


# In[4]:


print(predict[0])


# In[ ]:




