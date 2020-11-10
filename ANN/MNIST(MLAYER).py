# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:48:28 2020

@author: lenovoz
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("data/",one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y_true = tf.placeholder(tf.float32,[None,10])
pkeep = tf.placeholder(tf.float32)

layer_1 = 128
layer_2 = 64
layer_3 = 32
layer_out = 10
#truncated_normal ilk degerleri 0 yapmak mantıklı degil bundan dolayı
#Cok kücük rastgele sayılar atıyoruz.Bu fonksiyonda onu yapıyor.
#stddev ilede standartsapmasını belirliyoruz.bu sayede sayılar cok dagınık yerlesmiyor.
#burada yapılan uzunluk boyutu kadar bir vektor olusturup icine 0.1 koyuyoruz
W1 = tf.Variable(tf.truncated_normal([784,layer_1], stddev = 0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[layer_1]))  
W2 = tf.Variable(tf.truncated_normal([layer_1,layer_2], stddev = 0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[layer_2]))
W3 = tf.Variable(tf.truncated_normal([layer_2,layer_3], stddev = 0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[layer_3]))
W4 = tf.Variable(tf.truncated_normal([layer_3,layer_out], stddev = 0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[layer_out]))

#Buarada aktivayon gfonksiyonunu RELU sectik
y1 = tf.nn.relu(tf.matmul(x, W1)  + b1)
#droopout yaparak egitimde bazı nöronları uyutarak agın ayrıntılara cok odaklanmasını önlüyoruz
y1d = tf.nn.dropout(y1, pkeep)
y2 = tf.nn.relu(tf.matmul(y1d, W2) + b2)
y2d = tf.nn.dropout(y2, pkeep)
y3 = tf.nn.relu(tf.matmul(y2d, W3) + b3)
y3d = tf.nn.dropout(y3, pkeep)
logist = tf.matmul(y3d, W4) + b4
y4 = tf.nn.softmax(logist)

xent = tf.nn.softmax_cross_entropy_with_logits(logits=logist, labels = y_true)
loss = tf.reduce_mean(xent)

correct_pred = tf.equal(tf.argmax(y4, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
loss_graph = []
def traning_step(iterations):
    for i in range(iterations):
        #x_batch e resimlerin kendilerini,y_batch e ise gercek degerleri atanıyor.
        x_batch,y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x:x_batch, y_true:y_batch, pkeep:0.80}
        [_, train_loss] = sess.run([optimizer,loss], feed_dict=feed_dict_train)
        loss_graph.append(train_loss)
        if(i%100 == 0):
            acc = sess.run(accuracy,feed_dict = feed_dict_train)
            print("İterations :",i,"Accuricy :",acc,"Loss :",train_loss)

feed_dict_test = {x:mnist.test.images, y_true:mnist.test.labels, pkeep:1}
def test_accurcy():
    feed_dict_test = {x:mnist.test.images, y_true:mnist.test.labels, pkeep:1}
    acc = sess.run(accuracy, feed_dict = feed_dict_test)
    print("Testing Accurcy:",acc)

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def plot_example_errors():
    mnist.test.cls = np.argmax(mnist.test.labels, axis=1)
    y_pred_cls = tf.argmax(y4, 1)
    correct, cls_pred = sess.run([correct_pred, y_pred_cls], feed_dict=feed_dict_test)
    incorrect = (correct == False)

    images = mnist.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = mnist.test.cls[incorrect]

    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])

traning_step(5000)
test_accurcy()
#bu grafikte learning ratein uygunlugunu anlıyoruz.Eger Loss azalamıyorsa L_r buyuk diyebiliriz.
plt.plot(loss_graph,"k-")
plt.title("Loss Graph")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
plot_example_errors()