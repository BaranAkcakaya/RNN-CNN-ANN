# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:54:57 2020

@author: lenovoz
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data/",one_hot=True)
#place_holder = yer tutucu. Burada bir resim için yer açıyoruz.
x = tf.placeholder(tf.float32,[None,784])
#y_true resimlerin gerçek değerleri
y_true = tf.placeholder(tf.float32,[None, 10])
#Waigt and bias
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

logist = tf.matmul(x, W) + b
y = tf.nn.softmax(logist)
#logist burada tahmin oluyor. Burada softmax uygulanan vermiyoruz cunku
#bu fonksiyon zaten softmax uygulayacak. Ayrıca labels gercek degerleri alıyor.
#Aslında bu suna esit L(y^,y).Burada Loss fonksiyonunun matematiksel gösterimidir.
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logist, labels = y_true)
#bu method bize resim sayısı  kadar loss dönecek.Buda örn 500 reim varsa 500 olacak.
#Tabiki biz tum resimleri tek seferde almıyoruz.Bunun yerime parca parca alıyoruz.
#Her parcadaki resim sayısına bachsize diyoruz.Bachsize belirleyerek mesela her 100 resimde
#bir bize 100 tane loss döndürerek bunların ortalalamasını alacagız ve loss a yazacagız.
#burada reduce_mean o ise yarıyor
loss = tf.reduce_mean(xent)
#modelin ne kadar isabetli calıstıgını hesaplatıcaz.equal esit mi deye,argmax ise ektördeki
#en büyük sayının konumunu buluyor.
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
#burada ise gelen True yada False degerlerinini cast fonk kullanarak veri türünü degistirdik.
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#loss hesapladıgımıza göre GradientDescentOptimizer ile GradientDescent i optimize edecegiz.
#x+=learning_rate*dx
optimize = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
#Sess,on ,le kodu calıştırmak için oturum olusturuyoruz
sess = tf.compat.v1.Session()
#global_variables_initializer() bunu yapmayınca calısmıyormus.
sess.run(tf.global_variables_initializer())

batch_size = 128

def traning_step(iterations):
    for i in range(iterations):
        #x_batch e resimlerin kendilerini,y_batch e ise gercek degerleri atanıyor.
        x_batch,y_batch = mnist.train.next_batch(batch_size)
        feed_dict_train = {x:x_batch, y_true:y_batch}
        sess.run(optimize, feed_dict = feed_dict_train)
        
def test_accurcy():
    feed_dict_test = {x:mnist.test.images, y_true:mnist.test.labels}
    acc = sess.run(accuracy, feed_dict = feed_dict_test)
    print("Testing Accurcy:",acc)
    
traning_step(20000)
test_accurcy()















