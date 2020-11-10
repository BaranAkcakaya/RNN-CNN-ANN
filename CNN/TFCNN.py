# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:09:54 2020

@author: lenovoz
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#önceden resimleri düzleştiriyorduk.Bunu ondan false yapıyoruz.
mnist = input_data.read_data_sets("data/MNIST/", one_hot = True, reshape=False)

x = tf.placeholder(tf.float32, [None,28,28,1])
y_true = tf.placeholder(tf.float32, [None,10])

filt_1 = 16
filt_2 = 32

W1 = tf.Variable(tf.truncated_normal([5,5,1,filt_1], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1,shape=[filt_1]))
W2 = tf.Variable(tf.truncated_normal([5,5,filt_1,filt_2], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[filt_2]))
W3 = tf.Variable(tf.truncated_normal([7*7*filt_2,256], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[256]))
W4 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[10]))

#strides ile resim üzerinde gezen pencerenin özelliklerini belirliyoruz.
#strides = [batch, x, y, depth] dir.padding ilede resmin boyutunu koruyoruz
#ksize ise de pencere boyutlandırıyoruz
#Aktivasyon Bölümü,input=[28,28,1]
y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='SAME')+ b1)   #output=[28,28,16]
y1 = tf.nn.max_pool(y1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')     #output=[14,14,16]
y2 = tf.nn.relu(tf.nn.conv2d(y1, W2, strides=[1,1,1,1], padding='SAME')+ b2)  #output=[14,14,32]
y2 = tf.nn.max_pool(y2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')     #output=[7,7,32]
flattened = tf.reshape(y2, shape=[-1,7*7*filt_2])
y3 = tf.nn.relu(tf.matmul(flattened,W3) + b3)
logist = tf.matmul(y3,W4) + b4
y4 = tf.nn.softmax(logist)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logist, labels = y_true)
loss = tf.reduce_mean(xent)

correct_pred = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(5e-4).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
loss_graph = []
def traning_step(iterations):
    for i in range(iterations):
        #x_batch e resimlerin kendilerini,y_batch e ise gercek degerleri atanıyor.
        x_batch,y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x:x_batch, y_true:y_batch}
        [_, train_loss] = sess.run([optimizer,loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)
        if(i%100 == 0):
            acc = sess.run(accuracy,feed_dict = feed_dict_train)
            print("İterations :",i,"Accuricy :",acc,"Loss :",train_loss)

feed_dict_test = {x:mnist.test.images, y_true:mnist.test.labels}
def test_accurcy():
    feed_dict_test = {x:mnist.test.images, y_true:mnist.test.labels}
    acc = sess.run(accuracy, feed_dict = feed_dict_test)
    print("Testing Accurcy:",acc)


traning_step(5000)
test_accurcy()
#bu grafikte learning ratein uygunlugunu anlıyoruz.Eger Loss azalamıyorsa L_r buyuk diyebiliriz.
plt.plot(loss_graph,"k-")
plt.title("Loss Graph")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
