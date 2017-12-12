 #encoding=utf-8

import tensorflow as tf
import numpy as np
import time
from input import train, validation, test
batch_size = 50


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def conv2d_strides2(x,W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1],padding='VALID')

def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]
            
def squash(S):
    #S: shape [batch_size, 10, 1, 16]
    S_transpose = tf.transpose(S, [0, 1, 3, 2])
    S_length_square = tf.matmul(S, S_transpose)
    S_length = tf.sqrt(S_length_square)
    V = (S_length/(1 + S_length_square)) * S   
    return V

def routing(r, U_hat):
    #Note that the capsules in the lower layer select the capsules in the higher layers
    #U_hat: shape [batch_size, 10, 1152, 16]
    B = tf.constant(np.zeros([batch_size, 10, 1, 1152], dtype=np.float32))
    for i in range(0,r):
        C = tf.nn.softmax(B, axis = 1) #capsules in higher are candidates
        S = tf.matmul(C, U_hat)
        V = squash(S)
        U_hat_transpose = tf.transpose(U_hat, [0, 1, 3, 2])
        B = tf.add(B, tf.matmul(V, U_hat_transpose))
    return V

def loss(V, V_label):
    # V shape([batch_size, 10, 1, 16])
    # V_label shape([batch_size, 10])
    
    V_transpose = tf.transpose(V, [0, 1, 3, 2])
    V_length_square = tf.matmul(V, V_transpose)
    V_length = tf.sqrt(V_length_square)
    
    max_1 = tf.square(tf.maximum(0., 0.9 - V_length))
    max_2 = tf.square(tf.maximum(0., V_length - 0.1))
    max_1 = tf.reshape(max_1, [-1, 10])
    max_2 = tf.reshape(max_2, [-1, 10])
    
    summary = V_label * max_1 + 0.5 * (1 - V_label) * max_2
    L = tf.reduce_mean(tf.reduce_sum(summary, axis = 1))
    return L

def get_accuracy(V, V_label):
    V_transpose = tf.transpose(V, [0, 1, 3, 2])
    V_length_square = tf.matmul(V, V_transpose)
    V_length = tf.sqrt(V_length_square)
    V_length = tf.reshape(V_length, [-1, 10])
    correct_prediction = tf.equal(tf.argmax(V_length,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def loss_with_regular():
    return 0


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([9,9,1,256])
b_conv1 = bias_variable([256])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

W_conv2 = weight_variable([9,9,256,256])
h_conv2 = conv2d_strides2(h_conv1, W_conv2)

caps1 = tf.reshape(h_conv2, [-1,1,1152,1,8])
caps1 = tf.tile(caps1, [1, 10, 1, 1, 1]) 
caps1 = tf.transpose(caps1, [3,1,2,0,4])

W = weight_variable([1, 10, 1152, 8, 16])
U_hat = tf.matmul(caps1,W)
U_hat = tf.transpose(U_hat, [3,1,2,0,4])
U_hat = tf.reshape(U_hat, [-1, 10, 1152, 16])

V = routing(3, U_hat)
loss = loss(V, y_)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
accuracy = get_accuracy(V, y_)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(8400):
    batch = train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1]})
        print("step %d, traning accuracy %g"%(i, train_accuracy))
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#for i in range(200):
#    batch = validation.next_batch(50)
#    print("test accuracy %g"%accuracy.eval(feed_dict={
#        x: batch[0], y_: batch[1]}))

 
f = open('prediction.csv', 'w+')
f.write('ImageId,Label\n')
batchs = get_batchs(test, 50)
i = 1
for test_image in batchs:
    V_transpose = tf.transpose(V, [0, 1, 3, 2])
    V_length_square = tf.matmul(V, V_transpose)
    V_length = tf.sqrt(V_length_square)
    V_length = tf.reshape(V_length, [-1, 10])
    prediction = tf.argmax(V_length, 1)
    test_labels = prediction.eval(feed_dict={x: test_image})
    for label in test_labels:
        f.write(str(i) + ',' +str(label) + '\n')
        i += 1
        print("step % d"%i)
f.close()
