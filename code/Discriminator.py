import tensorflow as tf
import numpy as np

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.nn.Conv2d(3, 64, filters=(4,4),stride=(2,2), padding=1) # 64, 112, 112
        self.conv2 = tf.nn.Conv2d(64, 128, filters=(4,4),stride=(2,2), padding=1) # 128, 56, 56
        self.conv3 = tf.nn.Conv2d(128,256, filters=(4,4),stride=(2,2), padding=1) # 256, 28, 28, 2
        self.conv4 = tf.nn.Conv2d(256,512, filters=(4,4),stride=(2,2), padding=1)# 512, 28, 28
        self.conv5 = tf.nn.Conv2d(512,1, filters=(4,4),stride=(2,2), padding=1) # 1, 
        self.relu = tf.nn.LeakyReLU(0.3)

    def call(self, inputs):
        token = self.conv1(inputs)               #[-1, 64, 112, 112]
        token = self.relu(token)                 #[-1, 64, 112, 112]    
        token = self.conv2(token)                #[-1, 128, 56, 56] 
        token = self.relu(token)                 #[-1, 128, 56, 56] 
        token = self.conv3(token)                #[-1, 256, 28, 28]
        token = self.relu(token)                 #[-1, 256, 28, 28]   
        token = self.conv4(token)                #[-1, 512, 27, 27]
        token = self.relu(token)                 #[-1, 512, 27, 27]
        token = self.conv5(token)                #[-1, 1, 26, 26]
        return token