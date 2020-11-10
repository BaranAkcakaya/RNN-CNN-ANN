# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:56:25 2020

@author: lenovoz
"""
import os 
import re
import io
import requests
import numpy as np
import tensorflow as tf
from zipfile import ZipFile

data_dir = 'data/'
data_file = 'spam.txt'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
if not os.path.isfile(os.path.join(data_dir, data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')

    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1]      #sonuncu hariç hepsini alıyor

text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return (text_string)

text_data_train = [clean_text(x) for x in text_data_train]


max_sequence_lenght = 25    #max text uzunluğu
min_word_freq = 5           #minimum 5 kez geçmeli
embedding_size = 50         #her kerlime 50 boyutlu egitilebilir vektöre eklencek
rnn_size = 10

#burada kelimeleri vektörlere ekleyerek bu kelimelerin numaralarını vermis oluyoruz.
#bugun gunlerden salı,bugun gunlerden carsamba,vektor = [bugun,gunlerden,salı,carsamba]
#[0,1,2],[0,1,3] olur bu fonksiyon iste bu ise yarar
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_lenght, 
                                                                     min_frequency = min_word_freq)

text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

text_processed = np.array(text_processed)
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)

x_data = tf.placeholder(tf.int32, [None, max_sequence_lenght])
y_true = tf.placeholder(tf.int32, [None])

embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)

output = tf.transpose(output, [1, 0 , 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

w = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[2]))

logits = tf.matmul(last, w) + b
y = tf.nn. softmax(logits)

xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
loss = tf.reduce_mean(xent)

correct = tf.equal(tf.argmax(y, 1), tf.cast(y_true, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 250

def train_step (epochs):
    for epoch in range (epochs):
        shuffled_ix = np.random.permutation(np.arange(len(x_train)))
        x = x_train[shuffled_ix]
        y = y_train[shuffled_ix]

        n_batches = len(x) // batch_size + 1

        for i in range (n_batches):
            min_ix = i * batch_size
            max_ix = np.min([len(x), ((i + 1) * batch_size)])
            x_train_batch = x[min_ix:max_ix]
            y_train_batch = y[min_ix:max_ix]

            feed_dict_train = {x_data: x_train_batch, y_true: y_train_batch}

            sess.run(optimizer, feed_dict=feed_dict_train)

        [train_loss, train_acc] = sess.run([loss, accuracy], feed_dict=feed_dict_train)
        print('Epoch:', epoch + 1, 'Training accuracy:', train_acc, 'Training loss:', train_loss)


def test_step ():
    feed_dict_test = {x_data: x_test, y_true: y_test}
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print('Testing accuracy:', acc)

train_step(50)
test_step()




















