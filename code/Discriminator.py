import tensorflow as tf
import numpy as np
from Utils import IMAGE_SIZE

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (4,4), strides=(2, 2), padding="SAME") # 64, 112, 112
        self.relu1 = tf.keras.layers.LeakyReLU(0.3)
        self.conv2 = tf.keras.layers.Conv2D(128, (4,4), strides=(2, 2), padding="SAME") # 128, 56, 56
        self.relu2 = tf.keras.layers.LeakyReLU(0.3)
        self.conv3 = tf.keras.layers.Conv2D(256, (4,4), strides=(2, 2), padding="SAME") # 256, 28, 28, 2
        self.relu3 = tf.keras.layers.LeakyReLU(0.3)
        self.conv4 = tf.keras.layers.Conv2D(512, (4,4), strides=(1, 1), padding="SAME")# 512, 28, 28
        self.relu4 = tf.keras.layers.LeakyReLU(0.3)
        self.conv5 = tf.keras.layers.Conv2D(1, (4,4), strides=(1, 1), padding="SAME") # 1, 

    def get_model(self):
        input_ab = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 2), name='ab_input')
        input_l = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1), name='l_input')

        inputs = tf.keras.layers.concatenate([input_l, input_ab])

        token = self.conv1(inputs)               #[-1, 64, 112, 112]
        token = self.relu1(token)                 #[-1, 64, 112, 112]    
        token = self.conv2(token)                #[-1, 128, 56, 56] 
        token = self.relu2(token)                 #[-1, 128, 56, 56] 
        token = self.conv3(token)                #[-1, 256, 28, 28]
        token = self.relu3(token)                 #[-1, 256, 28, 28]   
        token = self.conv4(token)                #[-1, 512, 27, 27]
        token = self.relu4(token)                 #[-1, 512, 27, 27]
        token = self.conv5(token)                #[-1, 1, 26, 26]
        return tf.keras.Model([input_ab, input_l], token)

