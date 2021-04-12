import tensorflow as tf
import numpy as np
from .Utils import IMAGE_SIZE

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.VGG = tf.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        #self.VGG = tf.keras.models.Model(VGG_model.input, VGG_model.layers[-6].output)
        # Global Features

        self.global_conv1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')
        self.global_conv2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')

        self.global_conv3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')
        self.global_conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')

        self.flatten1 = tf.keras.layers.Flatten()
        self.global_dense1 = tf.keras.layers.Dense(1024)
        self.global_dense2 = tf.keras.layers.Dense(512)
        self.global_dense3 = tf.keras.layers.Dense(256)
        self.global_repeat = tf.keras.layers.RepeatVector(28*28)
        self.global_reshape = tf.keras.layers.Reshape((28, 28, 256))

        self.flatten2 = tf.keras.layers.Flatten()
        self.global_dense4 = tf.keras.layers.Dense(4096)
        self.global_dense5 = tf.keras.layers.Dense(4096)
        self.global_dense6 = tf.keras.layers.Dense(1000, activation='softmax')
        self.batchnorm = tf.keras.layers.BatchNormalization()

        # Midlevel Features

        self.mid_conv1 = tf.keras.layers.Conv2D(512, (3, 3),  padding='same', strides=(1, 1), activation='relu')
        self.mid_conv2 = tf.keras.layers.Conv2D(256, (3, 3),  padding='same', strides=(1, 1), activation='relu')

        # fusion of (VGG16 + Midlevel) + (VGG16 + Global)
        self.fusion = tf.keras.layers.concatenate
        
        # Fusion + Colorization
        self.out_conv1 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')
        self.out_conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')
        self.out_up_sample = tf.keras.layers.UpSampling2D(size=(2,2))
        self.out_conv3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')
        self.out_conv4 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')
        self.out_conv5 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')
        self.out_conv6 = tf.keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')

    def get_model(self):
        inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

        model_ = tf.keras.Model(self.VGG.input, self.VGG.layers[-6].output)
        vgg_res = model_(inputs)

        #global feature
        global_feat= self.global_conv1(vgg_res)
        global_feat = self.batchnorm(global_feat)

        global_feat = self.global_conv2(global_feat)
        global_feat = self.batchnorm(global_feat)

        global_feat = self.global_conv3(global_feat)
        global_feat = self.batchnorm(global_feat)

        global_feat = self.global_conv4(global_feat)
        global_feat = self.batchnorm(global_feat)

        global_feat2 = self.flatten1(global_feat)
        global_feat2 = self.global_dense1(global_feat2)
        global_feat2 = self.global_dense2(global_feat2)
        global_feat2 = self.global_dense3(global_feat2)
        global_feat2 = self.global_repeat(global_feat2)
        global_feat2 = self.global_reshape(global_feat2)

        global_featClass = self.flatten2(global_feat)
        global_featClass = self.global_dense4(global_featClass)
        global_featClass = self.global_dense5(global_featClass)
        global_featClass = self.global_dense6(global_featClass)

        #mid feature
        mid_feat = self.mid_conv1(vgg_res)
        mid_feat = self.batchnorm(mid_feat)
        mid_feat = self.mid_conv2(vgg_res)
        mid_feat = self.batchnorm(mid_feat)

        #fusion
        fusion_feat =  self.fusion([mid_feat, global_feat2])

        #output
        out = self.out_conv1(fusion_feat)
        out = self.out_conv2(out)

        out = self.out_up_sample(out)
        out = self.out_conv3(out)
        out = self.out_conv4(out)

        out = self.out_up_sample(out)
        out = self.out_conv5(out)
        out = self.out_conv6(out)

        out = self.out_up_sample(out)
        return tf.keras.Model(input=inputs, outputs = [out, global_featClass])